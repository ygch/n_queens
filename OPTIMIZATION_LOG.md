# CUDA N-Queens Solver: Optimization Log

## Baseline Performance

| Metric | Value |
|--------|-------|
| GPU | RTX 4090 (sm_89, 128 SMs) |
| N=20 kernel time | ~2329 ms |
| N=20 result | 39,029,188,884 |
| Registers | 20 per thread |
| Shared memory (N=20) | 5200 bytes (queens[64][20] + jkl_queens[20]) |
| Blocks per SM | 9 (smem-limited) |
| Warps per SM | 18 (9 blocks x 2 warps/block) |
| WG_SIZE | 64 (2 warps per block) |
| preset_queens | 6 |

NQueensFAF (OpenCL) reference: ~2491 ms. Our CUDA kernel is ~7% faster.

## Key Finding: The Kernel is Throughput-Bound

The kernel is **IPC-bound** (instruction throughput), not latency-bound. Analysis:

- RTX 4090: 4 warp schedulers per SM, max 4 instructions per cycle (IPC)
- 18 warps per SM, effective demand ~10.8 IPC (warps stall ~40% of time due to dependencies)
- Supply (4 IPC) << Demand (10.8 IPC) → throughput-limited
- **Increasing occupancy has no effect** — throughput is capped at 4 IPC regardless of warp count
- **The only way to improve is reducing total instructions per iteration**

The critical dependency chain `free -> queen -> ld -> free` (~30 cycles) is NOT the bottleneck because the scheduler interleaves other warps during stalls.

Total instructions estimate: ~15.2 per iteration (matches measured runtime at 4 IPC * 128 SMs * 2.5 GHz).

## Architecture

The solver uses:
1. **D4 symmetry group** — 8x reduction via rotation/mirror elimination
2. **Constellation method** — 4 border queens (i,j,k,l) + 6 preset queens, rest solved via GPU backtracking
3. **Jeff Somers bit-parallel backtracking** — ld/rd/col bitmasks, iterative DFS
4. **ld_mem/rd_mem** — captures bits lost during 32-bit shift operations (bit 31 of ld on `<<1`, bit 0 of rd on `>>1`)
5. **jkl_queens in shared memory** — per-row diagonal constraints from border queens, loaded once per block
6. **__funnelshift** — CUDA intrinsic for efficient 32-bit shift with overflow capture

## Failed Optimization Attempts

### 1. "Store free" with uint2 stack (kernel_free approach)
- **Idea**: Store `free_val ^ queen` (remaining free bits) alongside queen in `uint2` stack. On backtrack, load free directly instead of recomputing. Eliminates `direction` variable and anti-revisit mask.
- **Result**: 2906 ms (**26% slower**)
- **Why it failed**: Doubled shared memory (uint2 vs uint), halving occupancy from 38 to 26 warps. The dependency chain improvement didn't compensate.
- **File reference**: NQueensFAF `kernel_free.c`

### 2. "Store full free mask" in uint stack
- **Idea**: Store full `free_val` in a separate uint stack. On backtrack, load free directly.
- **Result**: 2977 ms (**28% slower**)
- **Why it failed**: Even though smem was reduced (3920 bytes, 48 warps), the extra `queen = -free_val & free_val` and `free_val ^= queen` computation on the backtrack critical path added instructions.

### 3. Smaller stack (N-5 instead of N)
- **Idea**: Stack only needs N-start-3 entries (start >= 3 for preset_queens=6). Use `queens[WG_SIZE][N-5]`.
- **Result**: 2498 ms (**7% slower**)
- **Why it failed**: Confirmed that occupancy increase (38->48 warps) has no effect on this throughput-bound kernel. The `row - MIN_START` indexing added minor overhead.

### 4. Manual 1x loop unrolling
- **Idea**: Duplicate the entire if/else body to reduce loop overhead and branch frequency.
- **Result**: 2521 ms (**8% slower**)
- **Why it failed**: Added 2 registers (20->22), increased code size. No throughput benefit since kernel is IPC-bound, not branch-bound.
- **Note**: NQueensFAF's OpenCL kernel uses 1x unrolling, but this doesn't help on CUDA.

### 5. "Col before free" (move col ^= queen before free computation)
- **Idea**: Move `col ^= queen` before `free_val` computation for the forward path. Since col now contains the queen's column, the anti-revisit mask `~queen` is redundant and can be eliminated. Saves ADD + LOP3 per forward iteration.
- **Result**: 2814 ms (**21% slower**)
- **Why it failed**: Making `col` depend on `queen` added it to the critical path for `free_val`. In the original, `col` was the old value (not dependent on queen), so the compiler could read it early and overlap with `ld` computation. The manual reordering constrained the compiler's instruction scheduling.
- **Bug**: Initial implementation also moved `col ^= queen` before `free` for backtrack/leaf paths, causing an infinite loop (queen bit re-appeared in free_val because `~(queen-1)` doesn't exclude queen when col no longer contains it). Fixed by keeping col after free for backtrack/leaf, but performance was still worse.
- **Lesson**: Don't manually reorder dependencies that the compiler handles better.

### 6. Bank conflict elimination via stride padding
- **Idea**: Pad `queens[WG_SIZE][N]` to `queens[WG_SIZE][N+1]` (odd stride, coprime to 32). Eliminates 4-way bank conflicts on `queens[lid][row]` access (with N=20, gcd(20,32)=4, threads 0,8,16,24 hit same bank).
- **Result**: 2332 ms (**0% change**)
- **Why it failed**: Bank conflict serialization (3 extra cycles per access) is hidden by the warp scheduler. Since the kernel is throughput-bound (not latency-bound), the extra cycles don't matter — other warps fill the gap.
- **Smem impact**: 5456 bytes (vs 5200), still 9 blocks/SM — no occupancy change.

### 7. Split epilogue (eliminate direction variable)
- **Idea**: Remove `direction` variable. Inline anti-revisit as `~queen` for forward and `~(queen-1)` for backtrack/leaf. Saves MOV + ADD per forward iteration.
- **Result**: Not benchmarked (reverted before testing)
- **Expected savings**: ~1.5 instructions per iteration (~10%)
- **Risk**: The compiler may already optimize `queen + direction - 1` via constant folding when `direction` is known at compile time within each branch. If so, no savings.

### 8. PTX inline assembly
- **Idea**: Write the inner loop in PTX inline assembly for precise instruction control.
- **Result** (from previous session): 7% **slower**
- **Why it failed**: PTX prevents the compiler from performing instruction scheduling optimizations. The compiler's scheduler is better than manual PTX at reordering independent instructions to fill pipeline bubbles.

### 9. 64-bit ld/rd (kernel_free_long approach)
- **Idea**: Use 64-bit `ulong` for ld/rd, eliminating ld_mem/rd_mem entirely. Saves 4 instructions per iteration (2 funnelshifts + 2 shifts for ld_mem/rd_mem management).
- **Result**: Not benchmarked directly (analyzed and rejected)
- **Why it failed**: 64-bit operations are emulated with 2x 32-bit on GPU. `(ld | queen) << 1` becomes 2 ORs + 2 shifts + carry handling (~4 instructions vs 2). Net instruction count INCREASES.
- **File reference**: NQueensFAF `kernel_free_long.c`

### 10. Occupancy increase (more warps)
- **Idea**: Increase warps per SM from 38 to 48 via smaller stack or fewer registers.
- **Result**: No improvement (actually slightly slower)
- **Why it failed**: Throughput is capped at 4 IPC by the warp schedulers. More warps don't increase throughput — they just increase scheduling overhead.
- **Confirmed**: The kernel is throughput-bound, not latency-bound.

### 11. Removing the ld_mem/rd_mem capture registers (faf kernel)
- **Idea**: The two funnelshift shift-registers cost ~4–6 SASS per iteration. The main-branch kernel runs to N=28 with plain shifts and no captures, suggesting the capture apparatus is vestigial.
- **Result**: Wrong counts. Removing the rd capture: N=10 → 788 (vs 724). Removing the ld capture: N≤14 accidentally correct, but N=20 → 40,219,954,082 (vs 39,029,188,884). **Both captures are load-bearing.**
- **Why it failed**:
  - rd: the bit shifted out at bit 0 (column N-1) looks harmless because `jkl_queens` blocks that column with `|1` — but a lost bit *cascades*: each unshift-restore shifts the deficient register left, so the missing bit migrates one position per restore, from the jkl-blocked column N-1 into interior columns → phantom candidates.
  - ld: a strand leaving through bit 31 (column N-32) is off-board *at that moment*, but backtracking moves it back into the board; the unshift-restore path cannot revive it once discarded.
  - Root cause: the capture registers are the price of the memory-lean 4B/level unshift-restore design. Only a reload-parent loop (full state per level) needs no captures — see attempt #12.
- **Evidence**: bisect variants (rd-only / ld-only removal), a Python replica that reproduces the first divergence (one frame: mem=0 vs plain=2 solutions), probe counters (N=10: rd restore-bit live 2553 times; ld restore-bit never).

## Successful Design Choices (in current code)

1. **CUDA port from OpenCL**: ~7% faster than NQueensFAF's OpenCL kernel
2. **preset_queens=6**: Optimal balance between CPU constellation generation and GPU backtracking (tested 4, 5, 6, 7)
3. **__launch_bounds__(64, 24)**: Controls register pressure, 20 registers per thread
4. **jkl_queens in shared memory**: Loaded cooperatively, accessed by all threads in block
5. **Constellation sorting by jkl group**: Groups constellations with same jkl for shared memory reuse
6. **Pseudo constellations**: Pad workgroups to multiple of WG_SIZE for aligned kernel launches (~1.9% waste)
7. **Multi-GPU support**: Ratio-based workload splitting across multiple GPUs
8. **__funnelshift for ld_mem/rd_mem**: Efficient 32-bit shift with overflow capture, maps to single SHF instruction
9. **Per-N kernel dispatch (reload for N<=21, capture for N>=22)**: see attempt #12 below

## Attempt #12: Reload-Parent Loop (successful for N <= 21)

**Design**: Replace the 4B/level queen stack + ld_mem/rd_mem unshift-restore with a 16B/level `int4` stack storing the full state `{cur, left, right, residual}`. Each iteration reloads the parent entry from shared memory (zero-bank-conflict layout: lane l's level e at slot `e*32+l`, 512B stride); no direction flag, no anti-revisit mask, no captures. Key identities (per-subproblem constants):

- leaf test: `popc(cur) == leaf_pc` with `leaf_pc = popc(col) + (N - start - 2)`; fires exactly at the node whose child row is N-2; count `+= popc(child)`
- jkl row: `jrow_base + popc(cur)` with `jrow_base = start - popc(col)`
- stack depth bound: `N - 3` (safe for every preset in [4, N-2], since start >= 1)
- dead roots (`valid == 0`) must return early — the original kernel effectively does the same (immediate pop)

**Validation**: The Python replica (`tools/nqueens_replica.py`) reproduces the original kernel exactly; the reload variant is 0-mismatch on 1747 real subproblem frames; all known totals N=6..22 are correct, including the deepest-stack edge cases with preset 4/5.

**Measured (RTX 4090, kernel ms)**:

| N | reload | capture (old) |
|---|---|---|
| 16 | 1.05 | 1.29 |
| 18 | 36.8 | 42.0 |
| 20 | 2172 | 2295 |
| 21 | 17393 | 18703 |
| 22 | 178.7 s | 162.5 s |

**Dispatch**: `N<=21` uses the reload kernel, `N>=22` keeps the capture kernel (`LAUNCH_N` / `LAUNCH_N_BIG` in `n_queens_cuda.cu`). Crossover at N=22 confirmed twice; the reload loses there because stack depth grows with N and shared-memory occupancy drops (17.4KB/block → 2 resident blocks).

## Attempt #13: Mid-Block V4 Architecture (evaluated on CPU + CUDA prototype, rejected)

**Idea** (a V4 variant): fix the two middle rows `m1=N/2-1, m2=N/2` as a V4-canonical queen pair (4x fold), pre-place k rows on the CPU outward from the middle, and let the GPU sweep the rest; claimed benefits: stronger pruning than border fixing, plus near-full reuse of the main-branch kernel.

**Findings**:
1. **The "upward traversal" form has a hidden correctness bug**: sweeping up from the middle drops diagonal strands that exit the (shift-register) board mid-sweep and re-enter during the lower sweep — brute force reproduced ghost placements (N=8, e.g. queens at (1,0) and (6,5) sharing a diagonal). The sound form is the standard top-down contiguous sweep; measured node counts of the two orders differ by only ~1%, so nothing is lost by using it.
2. The sound form reduces to: the faf kernel structure with a **generalized forced-row mechanism** — every fixed cell (pair + presets) gets a per-subproblem row-constraint table, forced rows get `~(cell bit)` entries, and below the last fixed row (`row > N/2`) the table is redundant (the carried masks already contain all fixed-cell constraints), so the deep loop is fully table-free.
3. **Exact node counts** (CPU anchored-representation counter, `tools/midv4_count.cpp`; solution totals verified exact at every N; candidate-event semantics identical to the GPU kernels):

| N | mid-V4 | faf | main | mid/faf |
|---|---|---|---|---|
| 14 | 7,462,218 | 6,021,910 | 13,463,757 | 1.239 |
| 16 | 307,581,550 | 239,628,220 | 563,126,914 | 1.284 |
| 18 | 15,823,729,869 | 11,884,168,803 | — | 1.332 |
| 20 | 998,190,890,794 | 728,354,432,000 | 1,865,473,917,695 | 1.371 |

The mid-V4 work is ~1.9x lower than main's but ~1.37x higher than faf's at N=20 (ratio grows ~+0.045 per +2N).

4. **CUDA prototype** (`src/mid_v4_proto.cu`): correct for all N up to 20 (39,029,188,884); best measured N=20: **4478–4506 ms** vs main 3636 ms vs faf 2172 ms.
5. **Optimization attempts all falsified the suspected bottlenecks**: split-table (deep region table-free) +1%; dynamic task queue no change in any correct configuration; occupancy 8→16 warps <2%; the LDS chain proven absent from 85–90% of passes with negligible effect. The remaining ~2x per-pass gap vs main at equal warp counts has no identified fix.

**Verdict**: rejected. The 1.37x node handicap vs faf (exact, unavoidable) cancels the kernel-reuse advantage; theoretical ceiling was parity-at-best, and the prototype lands below main. Kept for reference: `tools/midv4_count.cpp` (exact node counter), `src/mid_v4_proto.cu` (prototype).

## Attempt #14: Mid-Block V4 Decoupled Architecture (A0/A/B, ongoing, branch v4-mid-block)

Continuation of #13: the down phase was rebuilt as an independent kernel; full measurement campaign and parameter sweeps completed. Exact for N=8..22 (N=22 = 2,691,008,701,644).

### Architecture (vs the fused #13 prototype)
- **A0 count pass**: boundary states per subproblem -> prefix offsets (one launch)
- **A emit pass**: up phase (flipped-frame Somers + per-level cumulative lrc O(1) seam) -> global queue {cur, l2, r2, orbit-multiplier}
- **B consume pass**: flat down traversal, chunk=256 dynamic fetch; per-state loop inlined in PTX (next-state prefetch, multiplier packed in the slot, zero C wrapper)
- Queue streamed through VRAM in 4~72 batches (N=22 queue volume: 3347 GB)

### Key measurements
- **Decoupling itself = the single largest win**: N=18 223->76 ms, N=20 13.3->3.7 s (3.2~3.4x). A discriminating experiment proved the fused version's nesting cost scales with up-phase depth, not seam count; up and down MUST be separate kernels.
- **Flat per-pass rate**: down phase at 3.25~3.28 ps/pass for N=18/20/22 (average run 53->134->360 passes; amortization gain and tail loss cancel exactly).
- **Measured down-phase pass counts** (counter): N=18 15.7e9, N=20 995e9, N=22 **75.4e12**; N=22 boundary states 209.2e9.
- **ncu attribution** (N=18 B kernel): issue 74%, L1TEX 73.6%, avg active lanes **9.93/32** - the SIMT tax of short-run variance; the tool estimates +51% from fixing it, but no structural attempt realized it.
- **N=22 measured 267.1 s**: B=247.2 s + A=16.2 s + A0/misc 3.7 s. Same-machine: main ~245 s, faf ~180 s.

### Wins (committed)
| Optimization | Gain |
|---|---|
| Verbatim PTX port of main's loop + inlined per-state loop | +3~5% (N=18 +4%) |
| Next-state prefetch + multiplier packed in queue slot | +3% |
| CHUNK 256 (from 1024) | +2.7% (N=20) |
| BWG=96 block size (sole winner of the 64-256 sweep) | +2.5% (N=20) |

### All refuted directions (measured)
| Direction | Result |
|---|---|
| Length-capped checkpointing (cap interrupt + stack flush + ping-pong rounds, branch `v4-run-cap`) | run tails already bounded (only 0.002% exceed 512 passes); counter tax +14%; net loss |
| Register-carried stack top (zero LDS hot path; C and fully-predicated PTX) | 1.7~2.4x slower (backtrack becomes a real branch; warp serializes both paths) |
| Verbatim main-kernel clone (per-state atomic) | 4~9% slower (7.4e9 unamortized atomics) |
| Split residual array / 16B vector writeback | 15% slower / no effect |
| Occupancy 16->20 warps, DN_CAP shrink | no change |
| Grid sweep 512~4096 / very low grid 128/48 | flat / collapses 2.3~5.7x |

### Why the disadvantage grows with N (measured)
- mid's tree grows faster than main's: N=20->22 mid **x75.7** vs main x69.5 -> work advantage 1.87 -> **1.72** (deep boundary states x28.2 vs main seeds x1.4); extrapolates to -0.15 per +2 N
- flat rate ratio (1.74) plus a fixed ~7.5% phase tax (A/A0) -> +9% at N=22
- Multi-launch overhead audited: 145 launches < 1.5 ms; A0 re-traversal 3.5 s (1.3%); csum readback 0.3%; hiding A entirely (6%, 267->251 s) only reaches main parity - judged not worth pursuing

### Final positioning
Fastest at N<=18 (N=18: 63~66 ms, 3.5x faster than faf); parity with main at N=20 (3.50 vs 3.53 s); behind main by ~9% and faf by ~48% at N=22. **Short runs + the 4-fold V4 symmetry are architectural constants; no code-level lever remains.**

## Attempt #15: The Faster-Algorithm Search (map and closure, 2026-10)

Context: the NQueensFAF author's README claims a WIP method that is 40x faster single-threaded at N=22 and 100x at N=24 (~x2.5 per +2 N => per-row growth ~5 vs the DFS's ~8). A systematic search for a faster exact-counting algorithm was carried out; every route has a measured or mathematical verdict.

### Two new artifacts
1. **Closed form for the last two rows** (verified on 20,000 random states + a GPU micro-kernel):
   count2 = |V|*popc(F) - popc(V&F) - popc(V&(F<<1)) - popc(V&(F>>1)), with V = second-to-last-row candidates, F = last & ~(cur | (left<<1) | (right>>1)).
   Measured: correct but 40-52% SLOWER in this kernel (adds ~20 instructions per trigger vs the thin leaf layer it removes; adding instructions to the hot loop always loses). Cautionary note: a single reversed lop3 LUT (0xfe=OR vs 0x1=NOR) caused a 15x overcount, localized via an isolated micro-kernel test.
2. **Forward-window pruning family** (sound version with end-of-board geometry guard):
   | variant | N=16 nodes | vs baseline |
   |---|---|---|
   | plain DFS | 1.126e9 | - |
   | K=1 (skip if next row empty) | 0.726e9 | -35.6% = already the kernel's semantics (child==0 => no push) |
   | K=2 (skip whole placement if the relaxed window is empty) | 0.697e9 | only 2.6% beyond K=1, +5 instructions per placement => net negative on GPU |
   | K=3/4 | 0.807/0.990e9 | eaten by the geometry guard |

### Elimination map (all evidenced)
| family | verdict | evidence |
|---|---|---|
| symmetry folding | maxed | D4/V4 group limits |
| memoization/state merging | dead | X-ray near-injective (0.09-0.86% same-depth; tails too) |
| MITM/convolution joining | dead (math) | complementary-column-mask pairs ~ |A|^2/2^N >= solution count, then explode |
| frontier/profile DP | dead (math) | any sweep leaves one family global (row or column) => 2^N state |
| pruning (lookahead/window/arc-consistency) | dead | only 2.6% beyond the built-in K=1 |
| closed-form tail batching | correct but slow | see artifact 1 |
| structured inclusion-exclusion (diagonal-chain Mobius) | dead (math) | per-diagonal chain-connectivity state => 2^{2N}, same wall as DP |
| LGV/determinants | inapplicable | two diagonal families = double non-intersection; LGV handles one |
| displacement-counting (distinct differences) | inapplicable | that family is itself an exponential-enumeration problem; no joint closed form |

**Closure**: within known mathematical frameworks, exact n-queens counting provably has (1) a merge-free search tree (measured) and (2) no aggregation decomposition free of a global-family 2^N state (or 2^{2N} connectivity state). A "~5 per row" method therefore cannot come from these frameworks; the competitor's method must contain an unpublished new decomposition, or its 40x/100x figures fold in language/engineering constants (their old solver is Java).

## Attempt #16: GPU kernelization of MRV adaptive row order (measured negative, 2026-10)

Background: the #15 search map left exactly one live candidate — MRV (minimum remaining values) adaptive row order. On CPU (anchored representation) the node cut measured 2.05x @N=12 -> 2.75x @N=17 (+0.135 per +1 N). The anchored/seam conversion is exact (B1 bit = 2N-2-(c+r), B2 bit = N-1-(c-r); seam `B1=l2<<(N-1-RR)`, `B2=r2<<RR` lossless), so it was implemented end-to-end for an N=18 head-to-head.

### Three implementations — all exact, all slower (baseline: mid-V4 B kernel N=18 = 43.5-44.9 ms)
| impl | N=16 | N=18 | vs baseline |
|---|---|---|---|
| C v1 (slot smem recursive descent) | 12.5ms | 462ms | 10.5x slower |
| PTX v1 (SoA field layout, zero bank conflicts, per-visit reload) | 15.1ms | 692-718ms | 16x slower |
| PTX v2 (register-resident state, smem only on push/pop) | 28.6ms | 1059-1088ms | 24x slower |

(Results exact N=8..18; N=16 baseline 13.4ms.)

### Root cause: instruction economics (decisive)
- Baseline: 26-28 SASS per pass, fixed row order, no scan. MRV per node: scan m rows x ~20 instr (funnel+popc+argmin) + placement ~25 + push/pop 16 = 60-130 instr/node (m=1..8).
- To win, per-node cost must be <= node-cut ratio (2.75x); measured 5-8x -> structural loss.
- Occupancy deadlock: slot = 8 fields x 4B x 32 lanes = 1KB/warp/level x NLEV -> only 6 warps/SM resident; baseline stack 0.5KB/level. The register-resident variant (78 regs, zero spills) is still slower — third reproduction of space-map item 4.
- nvcc unrolls the state loop 7-8x, duplicating the huge asm; `#pragma unroll 1` neutral.
- Extrapolation: even with idealized scan cost, the node cut grows only +0.135/N -> net within +-10% through N<=26. Not a faster algorithm.

### Competitor status re-check (2026-10-11)
Upstream NQueensFAF latest commits end 2026-08 (dependency bumps only); the new method is still unreleased. Cross-calibration: their "N=22 single-thread 2h25m" ~ our stack's per-thread-equivalent (4090: N=22 = 267s; /30-60 thread-equivalents ~ 2.2-4.4h) — consistent with "40x vs their own old Java solver" = classical-class per-node efficiency + engineering constants, not an asymptotic breakthrough. If their "x2.5 per +2 N" ratio growth is real (~x5.5/row vs our x8.7-10), their N=28 edge is ~3-16x, not 40-100x.

## Attempt #17: Full baseline re-anchoring + 8-GPU production log archaeology (2026-10-11)

After #16, re-checked the real "beat faf" baseline. Several numbers in this log were stale ("main 245s / faf 180s / mid behind faf 48% at N>=22"). Re-measured everything today on the same 4090 (preset=6):

| N=22 | time | note |
|---|---|---|
| main (current HEAD; N=22 dispatches to capture kernel) | **162.2s** | count exact |
| faf-cuda original port (rebuilt from faf-cuda branch) | 163.5s | count exact |
| main forced to reload kernel at N=22 (temp build) | 179.8s | capture still 10.9% faster at N=22 (old "~9%" claim reproduced on the new build) |
| mid_v4 | 267s | still 3.5x ahead for N<=18 |

N=21: main 17.8s / faf 18.6s (main +4.3%).

**Conclusion**: current HEAD is at parity with faf at N=22 (0.8%, within noise) — the old "behind faf at N>=22" statement is void; ahead 4-19% for N<=21 and 3.5x for N<=18. "Faster than faf" is essentially already achieved by the current build (tie at 22); but no algorithmic progress toward the ~10x needed for N=28.
Root cause of reload's N=22 deficit (addendum): 16B/level smem traffic on push/pop vs 4B for capture = 4x the shared-pipe bytes; shrinking stack depth does not change bytes/transition (experiment: N-9 stack produced wrong counts immediately, and the gain path is structurally absent); combined with the "zero change" occupancy experiment, this route is closed.

### 8-GPU production log archaeology (logs/4090.log etc., 2025-08)
- Measured on 8 GPUs: N=22 37.4s, N=23 308s, N=24 2840s, N=25 7.9h (5090 cluster faster: N=22 29.1s).
- Single-GPU equivalent: that era's kernel was ~1.7x slower than today's (the 2026 kernel work delivered ~1.7x).
- **Static weight dispatch leaves 16-24% tail waste** (N=22: per-GPU 28.5->35.4s; N=25: 6h35m->7h52m). Dynamic re-dispatch by remaining work should recover most of it => ~1.15-1.2x production-side.
- Combined estimate: N=28 on 8x5090 with current kernels ~= 5.5-6.5 months (paper-era 11 months / 1.7 kernel / 1.15 dispatch).

## Attempt #18: Subproblem-level dynamic fetch (per-group atomic cursor) — measured neutral (2026-10-11)

Background: the only remaining kernel candidate from the space map's "not yet tried" list — dynamic work claiming, to remove the warp-tail waste (1.49-1.52x). Previously blocked by "the shared jkl table requires same-group workgroups". This experiment bypasses it with **per-group dynamic fetch**: each group (a run of equal-jkl subproblems; padWithPseudo guarantees WG_SIZE alignment) gets one global atomic cursor; lanes within a block claim the group's subproblems one at a time; blocks never cross groups, so the shared jkl table stays valid. Branch: `faf-dynfetch`.

### Results (all exact N=8..22; all timings neutral)
| comparison | static (v4-mid-block) | dynamic (faf-dynfetch) |
|---|---|---|
| N=20 | 2446ms | 2355-2454ms |
| N=21 (interleaved A/B x2) | 17823 / 17780ms | 17760 / 17831ms |
| N=22 | 162.2s | 162.4s |

### Diagnosis (new probe: per-subproblem iteration counters)
Static kernel, warp-level max/mean of subproblem iterations: **N=20 median 1.494** (p25 1.392 / p75 1.632 / p90 1.811), **N=21 median 1.448** — the imbalance is real (sortByJkl does not flatten it), **but balancing it yields nothing**.

### Conclusion (important correction to space-map item 1)
The inference "perfect balancing would take N=20 from 2172ms to ~1430ms" is **falsified** by this experiment: warp-tail length is not the bottleneck. Combined with every other flat result (occupancy 16->20, block/grid/chunk, instruction deltas, imbalance removal), the bottleneck is the **SIMT divergence structure itself** — per-lane independent deep-tree DFS, ncu measured only 9.93/32 active lanes (~31% useful lanes), and more resident warps buy only ~1.3x IPC. Beyond ~1.5x requires changing the execution model (warp-cooperative DFS — no viable design: deep trees have irreducible serial depth) or the search tree itself (#15, closed).

## Attempt #19: Profiling unlock (ncu) + two micro-fixes measured (2026-10-11)

Background: ncu was blocked by driver permissions until now, so all prior optimization was black-box. Unlocked this round: **kernel replay LaunchFailed on the huge-grid long kernels; `--replay-mode application` works** (plain sudo run suffices).

### Full profiles of both kernels (N=16, 4090)
| metric | reload (N<=21) | capture (N>=22) |
|---|---|---|
| IPC / issue slots | 1.85 / 46% | 2.79 / 70% |
| top pipe | ALU 43.5% | **ALU 75.4% (top)** |
| active lanes/32 | 17.3 (54%) | 11.4 (36%) |
| occupancy (achieved/theor.) | 24% / 29% (smem-limited 7 blocks) | 56% / 79% (smem-limited 19 blocks) |
| top stall | smem dependency 34% + ALU chain 33% | (no single dominant trigger) |
| smem per step | LDS.128+STS (~5-8 LSU cycles) | 4B (~1-2) |

### Two fixes implemented and measured
1. **Reload kernel: exact stack depth** (host computes maxd = (N-2)-start+1 over real subproblems; dynamic smem; `int S` param). Measured: N=16/20/21 interleaved A/B **neutral (+-2%)** — occupancy is not the reload's bottleneck (its limits are the 16B/step smem traffic and dependency chains). **Kept** (saves ~20% smem at large N; exact N=8..22).
2. **Capture kernel: queens stack shrink** ([WG][N] -> [WG][maxd+sentinel] + dynamic smem + odd stride vs bank conflicts). Measured: N=16 neutral; **N=22 = 176.6s vs baseline 162.2s (-9%) -> reverted**. Conclusion: an ALU-75%-bound kernel punishes any added address arithmetic; pushing occupancy past 56% buys nothing (24 resident blocks vs 17 equal-or-worse).

### Addendum (measured theorem)
Leaf semantics: across 400 real frames, leaf visits = solutions = sum of popc(free at leaf) = 29 — **free at the leaf is always single-bit**; bulk popc at the leaf saves nothing; and leaf-row iterations are only ~3.6% of all iterations (75.4e12 steps / 2.7e12 solutions ~ 28 steps/solution).

### Global verdict
The capture kernel (162.2s@N=22, production path) runs at **ALU 75% + issue 70% + 36% active lanes**; the faf-port (163.5s) sits at the same wall. Remaining micro-architectural headroom ~1.2-1.3x; the **36% lane efficiency (divergence tax) is structural** (per-lane independent deep-tree DFS) — the fundamental reason this kernel family cannot cross ~1.5x.

## Attempt #20: v3 register-carried kernel with 8B delta stack (refuted, 2026-10-11)

Design motive (from the #19 profiles): reload = smem-dependency heavy + low occupancy (LSU-bound); capture = ALU 75% + high occupancy. v3 takes the best of both: full register-resident state (zero smem traffic per step) + 8B/level {p, residual} dual-plane stack (stride-32, conflict-free) + u64 diagonal O(1) reconstruction on pop (no capture bits, no anti-revisit mask, no direction trick); semantics bit-exact to the reload kernel (leaf_pc / jrow_base formulas reused).

### Three bugs found and fixed during implementation
1. The bit0 dropped by `right >> 1` cannot be recovered by u64 (u64 rescues the <<1 high-bit loss only) -> added an mr path register (isomorphic to the capture kernel's rd_mem);
2. Pop-chain fusion (collapse consecutive exhausted levels in one trip) had the goto label after the result write (skipping it);
3. Comparison unit error: the C print multiplied by symmetry weights, making the trip counts look 13x apart; a Python replica cross-check confirmed the two algorithms have identical trip counts (200 frames: 1467 = 1467).

### Result: exact but uniformly slower
| N | v3 | reload | ratio |
|---|---|---|---|
| 16 | 151.6ms | ~110ms | 1.4x |
| 18 | 224.1ms | ~176ms | 1.3x |
| 20 | 5414ms | 2355ms | 2.3x |
(All counts exact N=8..22.)

### Root cause (structural)
The delta {p, residual} stack **cannot collapse exhausted levels**: the reload's 16B full-state snapshot lets a child take the slot of an exhausted parent (backtrack jumps straight to the grandparent); v3 must write an extra {p,0} entry and perform an extra pop-reconstruction per collapse -> trip count ~1.5-1.66x, which cancels the low-ALU/high-occupancy advantages.

Archived on branch `v3-regcarry`. **All three kernel designs (capture / reload / v3) are now implemented and cross-measured; the single-GPU kernel design space is closed.**

## Attempt #21: Dispatch boundary re-measured - N=22 flips to the reload kernel (-7%, 2026-10-11)

Background: the exact-depth stack from #19 (maxd=18 at N=22, dynamic smem) shrank the reload kernel's per-block smem from 19.5KB to 18.5KB, **crossing the resident-block threshold 4->5** (effective per-SM limit ~93-98KB), flipping the N=22 crossover. Full re-measurement:

| N | reload (exact depth) | capture | winner |
|---|---|---|---|
| 22 | **150.9 / 150.7 / 151.2s** | 162.2 / 162.0 / 161.96s | reload **-7%** |
| 23 | 1604.1s | **1469.7s** | capture -8.4% |

- Control: same source with the stack reverted to the static 19 levels = 179.9s (reproduces the old 179.8s) -> **the exact-depth change alone is -16% at N=22**, mechanism = resident blocks 4->5 (+25% warps).
- Growth: capture 22->23 = x9.06; reload 22->23 = x10.63 (the 5-block windfall disappears at 23; the reload is stuck at 4 blocks since 5 blocks would need >102KB at maxd>=20).
- Production dispatch now: **reload for N<=22, capture for N>=23** (commit 2c033a1; production binary re-measured 151.2s at N=22).
- N=24/25 boundary not measured (~3.7h each): capture expected to hold (reload pinned at 4 blocks); pin it later if needed.

## Attempt #22: Per-N workgroup-size table - second real speedup of the day (2026-10-11)

Motive: after the #21 "resident-block threshold" discovery, the block size directly changes per-block smem -> resident blocks and packing efficiency -> re-swept per N (using the fast-iterating N=19/20/21 plus an N=22 regression).

### Measured table
| N | best WG (reload) | measured |
|---|---|---|
| 16/17/18 | 32 ~= 64 (within noise) | - |
| 19 | 64 | 408ms (32:470 / 128:417) |
| 20 | **128** | **2195ms (-6.7% vs 2354 at 64)** |
| 21 | **32** | **16.72s (-6.0% vs 17.78 at 64)** |
| 22 | 64 | 150.9s (32:155.8; 128 predicted worse) |

Capture kernel: WG=96/128 beat 64 by 7-11.5% at two independent sizes (N=16, N=21) -> N>=23 launches at 128. Note the queens[MAX_WG][N] static array fixes the capture's per-block smem at 128*N*4 (14.3KB at N=28), capping resident blocks at 6-7 -> **a large WG is required to fill the warps** (WG64 = 12 warps, WG128 = 24-28). The 23+ choice is an extrapolation (that range is not measured by agreement); do one N=23 spot-check before production runs.

### Implementation (transferable to 23..28)
- Pseudo-constellation padding modulus 384 = LCM(32,64,96,128) so blocks never straddle a jkl group;
- launch_bounds(MAX_WG=128, 12) (same register budget as the old (64,24); codegen unchanged);
- Host picks the block size per N from the table; 23..28 uses 128.

All counts exact (spot-checked N=6..22); N=22 production re-measured 151.8s, no regression (commit c6b6162).

## Attempt #23: WG=128 spot-check at N=23 - extrapolation refuted (2026-10-12)

The #22 choice of WG=128 for N>=23 was an extrapolation (96/128 beat 64 by 7-11.5% at N=16/21). One direct N=23 spot-check per agreement:

| N=23 capture | time |
|---|---|
| WG=64 (old production) | 1469.7s |
| **WG=128 (new)** | **1476.9s** |

**Verdict: a tie (+0.5%, within noise) - the WG gain does not transfer to 23.** The WG effect seen below the crossover vanishes at this scale (and the capture kernel is not the production kernel down there, so its WG behavior differs). Count exact (24233937684440).

Decision: keep WG=128 for 23+ (measured tie vs 64; the [128][N] array puts 24-28 occupancy on par for both). **Scope of today's WG/dispatch gains: N<=22 only** (22: -7%, 21: -6%, 20: -6.7%); the 23+ region (the N=28 workhorse) gained nothing from this round.



## Attempt #24: Reload kernel manual unroll x2 (faf-audit byproduct, +1%, 2026-10-12)

While auditing faf's kernels.c I noticed it has a hand unroll-2 (duplicated loop body + mid-loop `row<start` exit) that our kernels lacked.

| kernel | N=19 | N=20 | N=21 | N=22 |
|---|---|---|---|---|
| capture unroll-2 | - | neutral (2521 vs 2527ms) | - | - |
| **reload unroll-2** | **+0.8%** | **+1.0% (2190->2169ms, 4 interleaved)** | **+0.5%** | **+1.2% (150.9->149.1s)** |

Verdict: unroll helps the reload only (small but consistent); shipped (commit c4ffa8e); the capture side is not adopted (neutral + unverifiable at 23+). Exact N=8..22.

### Same-day N=22 total: 162.2 -> **149.1s (-8.1%)** (dispatch flip -7% + unroll -1.2%)

## Attempt #25: kappa-model calibration + N=23 recaptured by reload-WG32 (-8%, 2026-10-12)

Finding: from the three measured points N=21/22/23, the "per-warp effective throughput" kappa = reload/capture ~= **3.6-4.0**, stable. Blind prediction for reload-WG32 at N=23: ~1350s -> **measured 1352.4s** (vs capture 1469.7s, **-8.0%**), prediction error 0.2%. Mechanism: the reload produces ~3.8x per warp; WG=32 packing gives it 9-10 resident warps (vs 8 at WG=64), crossing the flip point.

- Shipped: dispatch **23 -> reload @ WG=32** (commit a7474eb); N<=22 was already reload.
- **24-28 extrapolation (untested; no big-N tests by agreement)**: the kappa-model predicts reload-WG32 to beat capture by **10-16%** (warp arithmetic: N=24: 30/(9x3.85)~0.87; N=28: 24/(7x3.85)~0.89). Production keeps capture@128 for 24-28 (safe); **validate with one N=24 run (~3.5h) right before the N=28 launch, then switch** - trivial cost next to a months-long run.
- Same-day N=23 total: 1469.7 -> **1352.4s (-8.0%)**.

## Attempt #26: v4 register-carried + 16B snapshot stack kernel (4th refutation, 2026-10-12)

Design: "registerization" of the reload kernel - cur/left/right/residual resident in registers (zero smem access on sibling steps), full parent snapshot pushed on descent, pop = direct register overwrite; **exhausted-level collapse is free** (push only when parent residual!=0, pops land directly on the nearest non-exhausted ancestor - no reconstruction needed in the snapshot model). Goal: kill the per-trip LDS.128 dependency (the 34% short-scoreboard stall from the #19 profile).

Result: **exact on the first build** (N=8..18, the collapse design passed immediately), but slower:

| N | reload (production) | v4 |
|---|---|---|
| 16 | 114-117ms | 113-119ms (tie) |
| 18 | 167-175ms | 198-199ms (**+18% slower**) |
| 20 | 2165ms | 3944-3949ms (**+82% slower**) |

**Verdict: 4th refutation of the register-carried family** (prior: main-era "two-pass 3.3x slower / single-pass 1.6x slower", PTX inline -7%, v3 delta stack -1.4..2.3x). "The compiler schedules the dumb unconditional-reload loop better than hand register-residency" is now a law-level conclusion on this GPU/compiler. The reload kernel's per-trip reload IS the optimal form.



## Attempt #27: Register window (top-K levels into registers) - 5th refutation (2026-10-12)

Motive (direct reply to "can more registers replace shared memory?"): the N=21 profile shows 10.7% register usage vs 92% smem usage - the asymmetry is real. Design: **top-K stack levels resident in registers** (shallow levels, rarely visited), levels >= K stay in smem (hot path unchanged, index shifted by -K); expected smem 8.8KB -> 5.7-7.7KB -> blocks 10 -> 12-16 (+20-60% warps).

| variant | N=16 | N=20 |
|---|---|---|
| control (reload) | ~110ms | ~2210ms |
| K=6 (post spill fix) | ~111ms (tie) | **9181ms (4.1x slower)** |
| K=8 (post spill fix) | ~115ms | **10681ms (4.8x slower)** |

**Mechanism (decisive)**: the initial 16-29x blowup was first traced to **ptxas spilling the window registers to local memory** (42-reg budget from launch_bounds; 120-300B spills) - after relaxing to 128 regs and verifying zero spills, N=20 is STILL 4x. The real killer = the **warp divergence tax**: a warp's 32 independent subproblems are permanently spread across tree depths (21.8 active lanes at mixed depths), so the shallow and deep paths BOTH execute every trip (+ the variable-index switch branch trees + reconvergence) -> ~2x instructions per trip. The N=16 tie (small maxd, fewer boundary straddles) and the N=20 4x both fit this theory.

**Verdict (5th)**: registers-cannot-replace-smem on this workload is walled twice: (1) no dynamic register indexing in hardware; (2) any non-warp-uniform state split pays the divergence tax, and "32 independent deep trees per warp" makes uniformity impossible. The flat reload form is strengthened by a fifth independent line of evidence.

## Attempt #28: Within-group state sort (control-flow similarity) - refuted (2026-10-13)

Motive (user proposal): a warp's 32 lanes diverge at different tree points; if adjacent subproblems start from near-identical boards, branch sequences should correlate and divergence should drop. The current order only groups by jkl (generation order within).

Implementation: within each jkl group, sort by the starting-state proxy (col, ld, rd) lexicographically. Results (all counts exact):

| N | control (current order) | state-sorted |
|---|---|---|
| 16 | ~119.5ms | ~124.5ms (**+4.2%**) |
| 18 | ~169ms | ~190ms (**+12.4%**) |
| 20 | ~2159ms | ~2198ms (**+1.8%**) |

**Mechanism**: the generation order is already the strongest similarity clustering - each constellation's preset-expansion block is naturally contiguous (same i,j,k,l + adjacent preset rows); re-sorting by state values breaks those "siblings" apart and interleaves different constellations' expansions, lowering similarity net. **"Compute similar-path subproblems together" is already saturated by the enumeration order; classifier-style reorders cannot improve it.**

Addendum (closing "reduce if-branches"): unroll-2 is shipped (+1%, its mechanism exactly halving loop-branch density); leaf-inline / register family branch-reduction rewrites all measured negative (-2% and worse, 5 losses); unroll-4 estimated at only +0.3-0.5% with 2x code, abandoned.

## Attempt #30: Deep equivalence detection - different subproblems, equal counts (quantified closure, 2026-10-13)

Framework: a subproblem's count f depends only on the three mask sets (columns / two diagonal families); the systematically usable "different states, same count" = symmetry transforms acting on the three sets. Left-right mirror H: cur -> mirrored, left/right families swap with bit reversal.

Measured (main's real enumeration, incl. the mirror halving):

| detection layer | N=16 | N=18 | N=20 | rows=8 (N=18/20) |
|---|---|---|---|---|
| (1) identical set-triples | 0.48% | 0.31% | 0.21% | 1.05% / 0.74% |
| (2) H-mirror pairs (different states, equal counts) | 0.061% | 0.037% | 0.023% | 0.119% / 0.071% |

Observations: the rate rises with depth but caps at 1-2%; (2) is an order of magnitude below (1) (mirror redundancy is already quotiented at the placement level by the generator's halving; the residue is set-level coincidence).

**Structural ceiling**: systematic equal-counts = symmetry-orbit folding, bounded by the D4 order = 8x - already fully claimed by the production tree (faf border anchor); the main tree's top-row anchor only reaches the order-2 H subgroup (also fully claimed); mid-V4's middle anchor reaches order-4 (archived). Non-symmetric coincidences: the count is a #P-hard function of the masks with no algebraic identities; detecting a coincidence costs as much as computing the subtree. **Direction quantitatively closed: <1.2% combined.**

## Attempt #31: Strict D4 quota table for main's subproblem set (2026-10-13)

Question (user): without re-anchoring, apply D4/V4 folding directly to main's generated subproblems. Formalization: foldable amount = the set of sigma under which sigma(subproblem) is still a legal subproblem (= the stabilizer of the anchor).

| sigma | sigma(top-6-row prefix) | in the set? | foldable |
|---|---|---|---|
| H | top-6 prefix (row0 in 2nd half) | excluded by the generator's halving | **x2, fully claimed** |
| V / R | bottom-6-row anchor | No (theorem: set contains only top anchors) | 0 |
| M / A | rows = old columns | only when all cols <= 5 | **exactly 4 measured** (= 6x6-board solution count, verified at N=16/20) |
| odd-center trick | H-stabilizer case | - | fully claimed |

Complete anchor catalogue: top anchor stabilizer = order 2 (claimed); middle anchor = order 4 (mid-V4, claimed); border anchor = order 8 = full D4 ceiling (production tree, claimed). **D4 has only 8 elements; no method can exceed 8x.**

Production stack = main-lineage kernel (kappa 3.8x) x full-quota 8x border-anchor tree = 149.1s@N=22 (39% faster than main's 245s) - the user's goal's optimal attainable form IS the status quo.

## Attempt #32: Weighted-merge (orbit-quotient) theorem - formalizing the user's idea (2026-10-13)

Idea (user): merge main's subproblem set down to ~D4 scale, attach weights, multiply and sum.

Formalization: merging = choosing D4-orbit reps Q with weights w(q)=orbit size; requires the subproblem family to be D4-closed. Orbit-quotient size theorem: the number of "top-anchor members" in a generic orbit = |stabilizer of row-set {0..5}| = 2 (P and H(P)), and the halving already took one -> **M itself is already a section of the D4-orbit partition; #orbits ~= |M|, so in-place merging of main's set reduces nothing** (a mathematical consequence, not an algorithmic skill issue). The border family IS D4-invariant -> orbit quotient = 8x + weights 8/4/2 = **faf's constellation enumeration**.

Conclusion: the user's ideal combination (D4 scale + weights + fastest kernel) attains its optimum as the border-anchor tree x main-lineage kernel = **the current production stack** (149.1s@N=22 vs main 245s vs faf's kernel 163.5s). Beyond the status quo only three items: kernel (5 experiments all negative), folding (8x = D4 ceiling, claimed), or a brand-new decomposition outside D4 (absent in literature; competitor unpublished).

## Attempt #33: main-tree vs faf-tree work decomposition (measured closure, 2026-10-13)

Question (user): reload = main's kernel and the tree is ~1/4 the size, why did the time not drop to 1/4?

Measured (main branch binary rebuilt + faf-tree instrumented reload):

| quantity | main tree | faf tree (reload) | ratio |
|---|---|---|---|
| N=21 time | 29.0s | 16.5s (15.83s instrumented) | 1.76x |
| total nodes | ~11.6e12 (derived) | **6.096e12 (measured exactly)** | ~1.9x |
| per-node rate | ~2.42ps | **2.60ps (6.096e12/15.83s measured)** | 0.93 |

Check: 1.9 x 0.93 = 1.77 ~= measured 1.76. N=20: 3.65s vs 2.15s = 1.70x, same trend.

**Attribution**: (1) the "1/4" impression comes from the 8x fold, but the fold only acts on the ANCHOR ENUMERATION; total work is dominated by the shared deep sea below both anchors; (2) main's tree is NOT the unfolded faf tree - it is a different anchor family (top-6 rows) with its own 2x mirror halving, so 8x is not its baseline; (3) anchor-depth difference (6 rows vs 10 queens) grows with N (1.9 -> 2.0x). **Verdict: faf tree work ~= main/1.9; kernel rate at parity (main +7%, jkl cost tiny); end-to-end 1.7-1.8x (N=21: 29.0 vs 16.5s; N=22: 245 vs 149.1s).**

Addendum: the faf tree's N=21 total node count 6,096,149,831,979 matches the historical record digit-for-digit; the current build's rate 2.60ps vs the old build's 2.92ps (+11%) is a direct measurement of today's WG/exact-depth/unroll gains.

## Attempt #34: Branch cleanup - reload line moved to its own branch (faf-cuda-reload, 2026-10-13)

`v4-mid-block` had been carrying the faf reload production line (misnamed), with both kernels and experiment code in-tree. Split: the capture line stays on `faf-cuda` (original capture port, frozen), the reload line becomes its own branch.

Actions:
- New branch **`faf-cuda-reload`** from `faf-cuda` (b353e04): capture kernel + `LAUNCH_N_BIG` removed; **all N=6-28 dispatch to the reload kernel**; **WG for 24-28 changed 128 (old capture config) -> 32** (kappa-model prediction, NOT yet measured - validate before the first long run); everything else identical to the v4-mid-block @ 970c08d production state (unroll x2, exact-depth stack, WG table 20->128 / 21->32 / 23->32, NQ_WG override, pad 384).
- `faf-cuda` re-verified: builds; counts exact N=6-20 (N=20 = 2480.6ms) - frozen as the capture reference.
- Docs carried over (AGENTS / logs / CODE_PRINCIPLES / D4 / Q&A + tools/nqueens_replica.py); V4.md and the mid-V4 experiment files stay on v4-mid-block.

Validation (new branch production build): counts exact N=6-22 (N=6/7 need preset 4/5); N=20 = 2150.2ms, N=21 = 16.47s, N=22 = 147.95s - no regression vs v4-mid-block production (22 slightly faster than the 149.1s record).

## Measured Optimization Space Map (2026-10)

An instrumented campaign (per-thread iteration counters, warp-level analysis, occupancy queries, SASS dumps) bounded what remains after #12:

1. **Warp lane-tail waste = 1.52x — the biggest single loss.** A warp runs until its slowest lane finishes; measured warp-max/warp-mean of per-subproblem iteration counts: median 1.49. Perfect balancing would take N=20 from 2172ms to ~1430ms — but it needs a size oracle.
2. **No cheap size oracle exists.** Rank correlation with true per-subproblem iterations: start row -0.35; `popc(valid at start)` 0.14; 1-row expansion 0.14; 2-row 0.14; deeper frontier counts (n3..n5) decay to zero and go *negative* (wide shallow fans die early). Sorting by the best combination recovers only 2.6% of the waste (1.521 → 1.481). Dead roots at N=20: none.
3. **Utilization**: reload (10 warps/SM, 35 SASS/iter) ≈ 1.1 IPC/SM; capture kernel (32 warps/SM, 48 SASS/iter) ≈ 1.5 IPC/SM — i.e. 28–37% of the 4 IPC peak. Tripling resident warps buys only ~1.3x IPC.
4. **Register-resident loop variants** (state in registers, smem touched only on push/pop): two-pass (pop as its own iteration) 3.3x slower (warp divergence serializes both paths); single-pass with conditional save + popc-based row recompute 1.6x slower. The compiler's scheduling of the simple unconditional-reload loop wins.
5. **Deeper presets**: preset 7: 2358ms, preset 8: 2667ms (vs 2172ms at 6) — smaller subtrees have relatively larger variance, so the warp-tail waste grows.
6. **Remaining candidate (~+20% ± 10%, untested)**: per-thread dynamic task fetching (atomic queue) combined with register-computed jkl constraints — the shared-memory jkl table assumes same-group workgroups and is the structural blocker for dynamic fetching. Would convert most of the 1.52 waste into a single last-round tail, at the cost of ~5–7 extra ALU ops per iteration.

## Approaches Not Yet Tried

1. **ILP with 2 constellations per thread**: Each thread processes 2 independent backtracking searches, interleaving instructions. Likely neutral: chains per SM are set by smem, not by threads (see space map item 3).

2. **Persistent threads with global work queue**: Refined into "dynamic fetch + register-computed jkl" (space map item 6). The original plain dynamic fetch is blocked by the smem jkl table; V4 branch tested a similar scheme with per-row computed constraints (its jq block) at comparable per-subproblem cost.

3. **Different preset_queens for CUDA**: Now tested 4–8 on the CUDA kernels (incl. the reload kernel); 6 remains optimal, 7/8 clearly slower (space map item 5).

4. **Algorithmic changes**: Forward checking, constraint propagation, or bit-parallel pattern matching could reduce the search tree size. Would require significant rewriting; the space map suggests tree-size reduction is where the next >1.5x could come from.
