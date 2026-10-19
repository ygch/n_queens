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

## Attempt #35: Performance reproduction of the frozen faf-cuda line (2026-10-13)

Full re-verification of faf-cuda (original capture port) after the branch split (#34): build, counts and **performance**.

**Metric note: `CUDA kernel time:` = kernel only; the trailing `calc time` = end-to-end (includes CPU constellation generation/sort/transfer, dominant at small N). Historical tables (e.g. #12) list kernel times.**

| N | faf-cuda kernel time today (2 runs) | historical record (capture) | delta |
|---|---|---|---|
| 16 | 1.45 / 1.40 ms | 1.29 ms | +9-12% (abs <0.2ms, noise level) |
| 18 | 44.01 / 43.59 ms | 42.0 ms | +4% |
| 20 | **2293.0 / 2297.9 ms** | **2295 ms** | **+-0.1%** |
| 21 | 18610.4 ms | 18703 ms | -0.5% |
| 22 | **157.1 s** | 162.5 s | **-3.3%** (pristine slightly faster) |

- The 162.0-162.5s N=22 records came mostly from later builds (static `queens[MAX_WG=128][N]`, halved residency at WG64); the pristine build is hence slightly faster. Same-metric (end-to-end) today: reload@22 = 147.95s vs pristine capture 157.48s = **-6.1%** (the same-build A/B figure of -8.1% (149.1 vs 162.2) had the evolved capture penalty on the capture side).
- Small-N deltas (16/18) are sub-2ms jitter.

Verdict: the capture baseline is **fully reproducible** (N=20/21 are the hardest anchors, within 0.5%); faf-cuda is valid as the frozen capture reference.

## Attempt #36: N=24-28 branch comparison projection (exact maxd + kappa extrapolation vs the frozen capture line, 2026-10-13)

Refines the #25 kappa model for faf-cuda-reload (reload) vs faf-cuda (pristine capture): exact stack depths + the #35 correction that pristine `queens[64][N]` gives the capture more resident warps than the evolved `queens[128][N]` build.

Exact maxd (host program over real subproblems, same convention as host Step 3.6): **maxd = N-4** for N=22..28 (18/19/20/21/22/23/24; N=22 matches the measured 18; min_start=3, max_start=5 constant).

Projection (96KB effective smem; reload WG32 = 512B*maxd/block, capture WG64 = 260B*N/block; kappa=3.85):

| N | reload resident warps | capture resident warps | W_r*kappa | predicted T_r/T_c |
|---|---|---|---|---|
| 24 | 9 | 30 | 34.7 | 0.87 (-13%) |
| 25 | 9 | 30 | 34.7 | 0.87 (-13%) |
| 26 | 8 | 28 | 30.8 | 0.91 (-9%) |
| 27 | 8 | 28 | 30.8 | 0.91 (-9%) |
| 28 | 8 | 26 | 30.8 | 0.84 (-16%) |

Sensitivity: at 93KB effective smem the counts become reload 9/8/8/8/7 and capture 30/28/28/26/26 -> ratios 0.87/0.91/0.91/0.84/0.97: **reload leads at every N by -3% to -16% (median ~-10%)**. N=28 sits exactly on the reload 8->7 block boundary (worst case ~tie; WG64 is then equivalent, sweep it if validating).

Single-4090 absolute extrapolation (x9.1 per +1N, anchors at 23: reload 1352.4s / capture 1469.7s): 24 ~3.4h/3.7h; 25 ~1.3d/1.4d; 26 ~12d/13d; 27 ~107d/116d; 28 ~2.7y/2.9y (~2.5-3 months on 8x5090). **All unvalidated**; one N=24 reload run (~3.4h, predicted ~12.3ks) checks the whole ladder; full A/B ~7h.

## Attempt #37: warp-scaling law measured in the low-residency regime (NQ_WG squeeze, 2026-10-13)

Motive (answering "N=23 is already measured - is N=24 still needed?"): the only mechanism-level extrapolation in the 24-28 projection was "fewer resident warps at larger N (9-10 -> 7-8): does per-warp throughput hold?". An N=24 run does not cover it either (still the 9-block rung); but NQ_WG can squeeze N=21 (same work, same kernel, packing only) onto those rungs directly.

Measured (N=21 reload, same session, kernel time):

| WG | resident | warps/SM | measured | flat-rate prediction (T32*11/W) | ratio |
|---|---|---|---|---|---|
| 32 | 11x1 | 11 | 16275.5 ms | - (baseline) | 1.000 |
| 96 | 3x3 | 9 | 19548.6 ms | 19892 ms | **0.983** |
| 128 | 2x4 | **8** | 21972.1 ms | 22379 ms | **0.982** |

Verdict: **per-warp throughput is flat (even +1.7%) across 8-11 warps - no latency-hiding cliff.** The 8/7-warp rungs the reload naturally lands on at 26-28 are now directly evidenced; the last extrapolated mechanism in the #36 table is closed (worst case N=28 at 93KB/7 blocks -> ~0.95). This measures the scaling law (warps->time); kappa itself (~3.85) rests on the 21/22/23 blind test.

**Is N=24 still needed?** No longer a necessity for the conclusion: exact maxd (verified at 22/23) + the N=23 blind hit (0.2%) + today's measured scaling law cover the model. N=24's remaining value: (1) pre-N=28-launch sanity check (insurance, not required); (2) re-anchor the x9.1/N growth step once more; (3) a measured "capture@24" number for the paper/record if wanted.

## Attempt #38: N=24-28 re-projection from the paper's 8x5090 line (time-ratio ~= solution-ratio law verified, 2026-10-13)

Anchored on the paper's (main-line) 8x5090 measurements. The paper itself states the extrapolation law (p.9): "for N >= 23, the computation time ... increases by approximately 10 ... very close to the ratio of the number of solutions".

Law check (paper-line 8x5090 measured time ratios vs exact solution ratios):

| step | time ratio | solution ratio | time/sol |
|---|---|---|---|
| 23->24 | 9.48 | 9.39 | x1.009 |
| 24->25 | 9.92 | 9.70 | x1.022 |
| 25->26 | 9.87 | 10.11 | x0.977 |
| 26->27 | 11.36* | 10.53 | x1.079 |
| 23->27 total | x10545 | x9693 | **x1.088 (~+2.1%/step)** |

(*26->27 includes the preset 6->7 change and multi-GPU imbalance; 25->26 has 2.5d rounding.) Law holds within +-2-8% per step, +8.8% cumulative.

**Paper line, 8x5090, N=24-28**: 24 = 2205s (36.8min), 25 = 21880s (6.08h), 26 = 2.5d, 27 = 28.4d (all measured); **28 = 28.4d x 10.9 (trended solution ratio) x (1.00-1.08) = 310-335d ~ 10.2-11.0 months** (the paper's own estimate: 340d = 28.4x10.52x1.14; a(28) unknown, ratio trended +0.4/step).

**Current line (faf-cuda-reload), 8x5090** (divide by 1.76-1.85 from the #33 end-to-end measurement; cross-checked with the reload self-chain): 24 ~ 21min, 25 ~ 3.4h, 26 ~ 1.4d, 27 ~ 15.5-17d, **28 ~ 5.5-6.5 months (~6 center)**.

**Correction to #36**: the flat x9.1/N extrapolation understated 26-28 (solution ratios rise to 10.1-10.9); N=28 on 1x4090 goes from "2.7y" to ~5.0-5.8y. Also: the paper itself notes load balancing is not optimal at large N (5-day GPU finish spread in the 27 run); fixing it saves ~10% more.

## Attempt #39: Static vs dynamic shared memory (equal-size isolation A/B, 2026-10-13)

Motive (user question): the paper (main line) found static shared memory faster than dynamic, while the reload line uses dynamic - run an equal-byte-size isolation experiment (residency eliminated; only "compile-time static size vs extern dynamic base" differs).

Method: reload kernel changed to `__shared__ int4 stk[(WGW/32)*(N-4)*32]` - with preset=6, maxd=N-4 is known (verified), so the static size is exactly equal to the dynamic allocation's bytes; launch passes no dynamic smem; a host assertion `maxd <= N-4` guards preset!=6. N=20/21/22 x WG32/64/128, interleaved A/B, 2 runs each:

| config | dynamic (production) | static (experiment) | delta |
|---|---|---|---|
| 21@WG32 (11 warps) | 16299.6 ms | 16292.1 ms | -0.05% |
| 21@WG64 (10 warps) | 17271.5 ms | 17283.8 ms | +0.07% |
| 20@WG128 (12 warps) | 1986.1 ms | 1985.6 ms | -0.02% |

(Counts exact on both: 314666222712 / 39029188884.)

**Verdict: completely insensitive (+-0.1%, within noise).** The static/dynamic difference is a one-time base materialization; irrelevant to the reload's LDS-dependency bottleneck. The paper's "static faster" was specific to the main-line row-stack configs (its dynamic side also varied carveout/L1 and block size); the reload's earlier "static-19 -> exact-depth dynamic = -16% @N22" win came entirely from **resident block count** (19.5 -> 18.5KB crossing 4->5 blocks), not the allocation mechanism - this experiment closes the loop.

Engineering call: **production stays dynamic** (static deployment would need per-(N,WG) instantiations, 3x instantiation count, and hits the 48KB static cap at large N - for zero gain).

## Attempt #40: reload-vs-capture gap shrinks with N - recheck and 24-28 projection correction (2026-10-13)

User observation ("does the reload lead shrink as N grows?") confirmed. Pristine-capture reference (calc metric, 1x4090):

| N | capture | reload | reload lead |
|---|---|---|---|
| 19 | 443 ms | 396 ms | 10.6% |
| 20 | 2490 ms | 2150 ms | 13.6% |
| 21 | 18.77 s | 16.49 s | 12.2% |
| 22 | 157.5 s | 148.0 s | 6.1% |
| 23 | ~1425 s (est.) | 1352.4 s | ~5.1% (-8.0% vs the evolved capture) |

**11-14% (19-21) -> 5-8% (22-23): the lead roughly halved, beyond noise.**

Mechanism: lead ~= 1 - W_c/(W_r*kappa). Both sides' resident warps step down with N (per-block smem ~ N); from N=22 the reload's rung drops first (12->10->9 warps) while the capture holds a 30-34-warp plateau. Per-row effective kappa back-fit from the 19-23 pristine pairs: **3.2-3.6** (3.2-3.4 with the +1KB/block reservation accounting; 3.4-3.7 without) - **#36's kappa=3.85 mixed the two accounting conventions and is 10-20% too high**.

**24-28 correction**: recomputing the #36 ladder with kappa=3.2-3.6 (full enumeration) gives T_r/T_c = **0.93-1.06 (-7%..+6% near-tie band, center ~-2..3%)**, retracting the "reload wins 13-16%" claim; in the most conservative conventions the capture can edge ahead 1-5% at individual N. Absolute-time predictions unchanged (reload@24 ~ 3.4-3.5h).

**Impact**: branch choice unchanged (the current branch has reload only); **the N=24 run regains its value as adjudicator** of "shrinkage continues vs ladder rebound" and of the true kappa (reload 3.4h; full A/B ~7h).

## Attempt #41: Re-calibrated 8x5090 estimates for N=24-28 from the 16-PPU measurements (with the systematic main config-downgrade penalty, 2026-10-17)

Context: the user validated all three lines on 16 PPUs (counts exact; logs+xlsx in the repo root) and provided two facts: (1) the PPU main build has PTX replaced by plain CUDA (~7% slower); (2) main's 26->27 step dropped one configuration, costing >10% beyond the solution-ratio growth, and 27->28 will pay another >10% - a systematic penalty, not noise.

**PPU -> 8x5090 calibration (two independently verifiable factors)**
- reload line: PPU / 8x5090 = **1.26** (three points: 169.6/132.2=1.28, 1586.7/1253=1.27, 15409/12292=1.25);
- main line: PPU(no-PTX) / 8x5090 = **1.62** (N=23: 376.4s / 1.62 = **232.3s vs the paper's measured 232.7s - 0.2%**; includes the 7% PTX loss and a 1.51 hardware factor);
- cross-check: the PPU is 1.28x more favorable to reload than to main (1.62/1.26), consistent with reload beating capture by 1.8-1.9x on PPU.

**Law check (PPU)**: 23->24 = x9.36 (solution ratio 9.39), 24->25 = x9.71 (9.70) - zero drift; extrapolate directly with solution ratios.

**Re-estimate (8x5090)**

| N | reload line (rows=7) | main line |
|---|---|---|
| 24 | 1259s = 21.0min | 2205s = 36.8min (measured) |
| 25 | 12229s = 3.40h | 21880s = 6.08h (measured) |
| 26 | 1.43d | 2.5d (measured) |
| 27 | 15.1d | 28.4d (measured; includes the >=10% config penalty) |
| 28 | **164d ~ 5.4 months** | **~340d ~ 11.2 months** (28.4d x 10.9 x 1.10; matches the paper's 340d = 28.4 x 10.52 x 1.14) |

**Correction to #38**: main@28 should not use "solution ratio x 1.00-1.08 drift" (10.2-11.0 months) - the config downgrade is a systematic >=10% term per step; with it, ~11.2 months. The reload line has no config wall (block=32, 12.3KB/block at N=28), so it keeps the clean solution-ratio extrapolation: **N=28 ~ 5.4 months**. The gap between the two lines widens to **~2.1x** at N=28.

(16-PPU equivalents = the table x1.26: 24=26.5min, 25=4.28h, 26=1.8d, 27=19.0d, 28~6.8 months - matching the measured 24/25 exactly.)

## Attempt #42: Cross-hardware check of preset rows (N=22 A/B) and the parameter status for N=24-28 (2026-10-17)

Motive: on 16 PPUs, rows=7 beat 6 by 8-17% for N>=22, while the reload branch defaults to rows=6 - does it transfer to NVIDIA?

Exact stack depth (host program, preset 6 vs 7): preset7 sets min_start 6->4, so **maxd = N-5** (one slot less per warp: saves 512B; may add one resident block at N>=26); subproblem counts scale x9.8-15.3.

**N=22 A/B (single 4090, calc)**: rows=6 = **151.77s**, rows=7 = **164.05s (+8.1% slower)** - consistent with the old N=20 test (6 faster by 8.6%). Verdict: **the rows=7 gain is PPU-specific and does NOT transfer to NVIDIA**; keep rows=6 on 4090/5090.

Parameter status (reload branch):
- **preset rows=6**: locked for NVIDIA (N=20 and N=22 evidence); PPU uses 7 (measured optimal there).
- **block size 24-28 = 32**: model-projected, NOT measured at 24+ (N=23 measured -16% supports it; at 27-28 the model predicts 8 vs 6 warps for 32 vs 64).
- **Multi-GPU mod split**: verified on 16 PPUs (exact counts, 0.4-1.2% finish spread).
- Exact-depth stack / unroll x2 / padding 384: verified.
Remaining optional check: block 32 vs 64 at N=24 (NVIDIA ~7h for both; PPU ~26.5min each).

## Attempt #43: optimal preset shifts up with GPU count (5->6->7) - correcting #42's wording (2026-10-17)

User correction (with existing evidence): the optimal pre-placement rows increase monotonically with the GPU count - single GPU optimum = **5** (paper README: "single GPU, N<=22: pre-placing first 5 rows with configuration3"); 8 GPUs (4090/5090) = **6** (all paper 8-GPU tables use preset 6/config2); 16 PPUs = **7** (measured, N>=22). This is not PPU-specific.

Mechanism: more GPUs -> fewer subproblems per GPU -> larger relative variance of per-GPU workload -> finer decomposition needed; each extra preset row multiplies the subproblem count by x10-15 and makes trees smaller/more uniform -> better balance and (on PPU) better per-node efficiency. Single GPU is the opposite (fewer, longer trees minimize per-subproblem overhead) - optimum stays at 5-6.

**Correcting #42's conclusion**: the N=22 single-4090 A/B (rows6=151.8s vs rows7=164.1s) only shows that the *single-GPU* optimum is <=6; it does NOT imply 7 is bad on 8x5090 - the multi-GPU optimum is set by the GPU count. Action:
- single 4090 (reload line): 6 (4-8 measured; 5 is the main+config3 result);
- 8x5090: 6 (paper-validated);
- 16 PPU: 7 (measured). #41's calibration (PPU@7 / 5090@6) was already optimal-vs-optimal; numbers unchanged.

## Attempt #44: reload multi-GPU measurements on 8x4090 (N=20-23, rows 6/7, 2026-10-17)

User measurements (second multi-GPU platform besides 16 PPUs):

| N | reload r6 | reload r7 | paper 8x4090 (main r6) | single 4090 (reload r6) |
|---|---|---|---|---|
| 20 | **2.19** | 4.04 | 2.58 | 2.15 |
| 21 | **4.19** | 7.49 | 6.16 | 16.49 |
| 22 | **22.3** | 27.2 | 37.46 | 148 |
| 23 | **184.7** | 194 | 308.5 | 1352.4 |

Conclusions:
- **rows=6 wins everywhere on 8x4090** (r7 is 5-84% slower) - the 8-GPU tier of the "preset rises with GPU count" rule is 6 (matches the paper; 7 is the 16-PPU tier).
- **Multi-GPU speedup vs single 4090**: 0.98x / 3.94x / 6.64x / 7.32x. N=20/21 is swamped by a **~1.8s constant** (8 CUDA contexts/module init + generation + gather/copies): total ~= kernel/8 + 1.8s. At N>=22 the compute-only scaling is 7.3/8 = **92%**.
- **vs paper's 8x4090 (main)**: 1.67-1.68x faster at N>=22 (matches the single-card line ratio 1.66-1.73x); N=20 only 1.18x - both sides swamped by the ~2s overhead.
- **Independent cross-check of the 8x5090 estimate**: 8x4090 x 0.79 (mean of the paper's eight 8-GPU 5090/4090 points): N=23 ~ 146s vs #41's ~132-135s - within 8-10%.
- **16 PPU ~= 8x4090** (3-17% apart at N<=22; PPU 169.6s at N=23 is 9% faster) - consistent with "each PPU ~ 0.5x a 4090"; the PPU r7 gain starts to show at N=23.

## Attempt #45: preset reversal at N=24 on 8x4090 (r7 wins) - the residency-step mechanism (2026-10-17)

New measurements (8x4090, block=32):

| N | r6 | r7 | winner |
|---|---|---|---|
| 20 | 2.19 | 4.04 | 6 |
| 21 | 4.19 | 7.49 | 6 |
| 22 | 22.3 | 27.2 | 6 |
| 23 | 184.7 | 194 | 6 |
| **24** | **1867** | **1733** | **7 (-7.2%, reversal)** |

Mechanism: preset 6->7 reduces maxd by exactly 1 (N-4 -> N-5, saving 512B per warp). Resident blocks at WG32 under the 96KB-effective-smem convention (#36):

| N | maxd 6/7 | blocks 6/7 |
|---|---|---|
| 23 | 19/18 | 10/10 |
| **24** | **20/19** | **9/10 <- steps diverge** |
| 25 | 21/20 | 9/9 |
| 26 | 22/21 | 8/9 (next divergence) |
| 27 | 23/22 | 8/8 |
| 28 | 24/23 | 8/8 |

N=24 is the first divergence: r6 falls from 10 to 9 blocks (-1 warp/SM ~ -11%), r7 keeps 10 -> reversal. Matches the measured flip point. Cross-check with the solution law: r6's 23->24 = x10.11 (+7.7% over 9.39), r7's = x8.93 (-4.9%) - "one falls off a step, the other doesn't".

Updated guidance:
- 8x4090: **r6 for N<=23, r7 for N>=24** (model: tie at 25, r7 again at 26, ~tie 27-28);
- the N=22 flip on 16 PPUs is the same mechanism shifted by a different smem budget/hardware - #43's "rises with GPU count" stands, with the mechanism clarified as maxd residency steps;
- 8x5090 (same arch family): likely r7 for 24-28 as well - **the N=24 8x5090 estimate moves from 21min (r6) to ~21-23min (r7)**;
- the two anchor chains differ ~9% at N=24 (PPU 1586.7s vs 8x4090 1733s; the 16-PPU rig is relatively stronger at large N).

## Attempt #46: block-size A/B at N=24 on 8x4090 (32 vs 64) - 32 wins (2026-10-17)

User measurement (8x4090, N=24, rows=6, calc):

| block | time | delta |
|---|---|---|
| 32 (production default) | 1867 s | - |
| 64 | 1968 s | **+5.4% slower** |

Conclusion: **the block=32 projection for 24-28 gets its first direct measurement support at N=24** (previously only the N=23 -16% and the ladder model). The per-N table stays 24-28 -> 32.

Note: measured at rows=6; with rows=7 (maxd=19) the 32/64 residency is closer (10 vs 10 warps under the 96KB convention), but 32 is already the better side - and the fastest N=24 combo (r7 + block32 = 1733s) uses 32 anyway.

## Attempt #47: r6/r7 share the same imbalance source on 8x4090 - per-card speed differences (2026-10-17)

Exact accounting of the two N=24 runs on 8x4090 (user logs):

| run | spread | spread/mean | **slowest/mean (real loss)** |
|---|---|---|---|
| r6 (1867s) | 107.7s | 5.88% | **+1.94%** |
| r7 (1733s) | 78.9s | 4.68% | **+1.94%** |

**Correction**: the earlier inference that "1.5-1.9 points of r7's lead come from balance" is falsified - the 7.2% lead comes entirely from the residency step (#45); the imbalance loss is unchanged.

Per-card relative durations (both runs, deviation from each run's mean):

| gpu | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| r6 | +0.6% | +0.7% | +1.5% | -2.1% | +1.0% | +1.9% | +0.4% | **-3.9%** |
| r7 | -0.5% | +0.1% | +1.6% | -1.1% | -1.0% | +1.6% | +1.9% | **-2.7%** |

Highly consistent across runs (gpu7 always fast, gpu2/5 always slow) -> the residual imbalance is dominated by **persistent 2-4% per-card speed differences** (clocks/thermals/power/slots), not workload granularity: r7's 11x more samples did not smooth it. Contrast: the homogeneous 16-PPU server shows only 0.49% spread at r7.

Recommendations: (1) check `nvidia-smi -q -d CLOCK,POWER` during a run to rule out thermal/power issues (free ~1.9%); (2) if it is genuine card variance, weighted per-card assignment can recover up to 1.9% (not worth it on the 16-PPU production rig, whose spread is 0.4%); (3) a cross-GPU dynamic queue is the theoretical fix but not worth 2%.

## Attempt #48: row-identity simplification (removing the per-iteration POPC) - refuted by stack collapse (2026-10-17)

Motive (fresh review): if `jrow_base + popc(cur)` is identically `start + top`, one POPC per iteration can be removed and the jkl LDS address dependency becomes an IADD instead of a POPC (ncu: 33.9% ALU-chain stalls).

Derivation: the entry at depth i stores cur with exactly `init_pc + i` bits -> `popc(cur) = init_pc + top`; row = `start + top`; the leaf test becomes `start + top == N-2`.

**Measured: compiles, but counts are wrong for N=8-20** (N=8: 76 vs 92; N=20: 124408 vs 39029188884).

**Root cause (important structural fact)**: the reload stack **collapses** - when a parent's `residual == 0`, `top--` lets the child overwrite the parent's slot (the "child replaces exhausted parent" optimization from #20). Collapse decouples the entry index from the depth, so the identity fails after any collapse. => **the stored POPC is the necessary price of the collapse optimization** and cannot be replaced by index arithmetic.
Reverted (source restored to HEAD, binary restored; N=19 verified). Direction closed.

## Attempt #49: measured accidental equivalence in the reload tree - nothing left after the D4 quotient (2026-10-17)

Computational-equivalence pairs in the faf-reload constellation set (state = `(ld, rd, col, start, j, k, l)` identical <=> residual subtrees mergeable):

| N | pairs/elements | share | vs main tree (#29) |
|---|---|---|---|
| 20 | 90 (180) | 0.0131% | 0.21-0.48% |
| 24 | 152 (304) | 0.0053% | - |
| 28 | 230 (460) | 0.0025% | - |

Verdict: after the D4 quotient the residual equivalence is 1-2 orders of magnitude below the main tree (main never did the D4 quotient; its duplicates included D4-class pairs) and decreases with N. Merging everything saves <=0.01%; deep-state memoization is even less worthwhile on the reload tree. "No residual symmetry/equivalence after D4" holds more strongly for the reload branch.

## Attempt #50: N=25 (rows=7) on 8x4090 = 18,384s - 9.3% above the naive solution-ratio extrapolation, but exact under the residency-step model (2026-10-17)

Measured: 8x4090, N=25, rows=7, block=32 -> **18,384 s (5.11 h)**. The earlier "1733 x solution ratio 9.70 = 16,817s (4.67h)" extrapolation **missed the step**:

- preset7 maxd: 19 at N=24 -> **10 resident blocks** (10 warps); 20 at N=25 -> **9 blocks (-1 warp ~ -10%)**;
- ladder-model prediction = 1733 x 9.70 x (10/9) = **18,686 s**; measured 18,384 (**-1.6%**) - matches;
- this also falsifies the +1KB/block-reservation convention (which predicts no step at 25) and supports the 96KB no-reserve convention (#45).

Action: at N=25, r6 has maxd=21 -> still **9 blocks** (tied with r7) -> r6 predicted 18,117s, plus the "r6 ~3-5% faster in tie cases" (N=23 evidence) -> **~17.2-18.1k (2-6% faster than the r7 measurement)**. **Prefer rows=6 at N=25** (untested; worth verifying if re-running).
Updated 8x4090/5090 preset table: **24->7, 25->6, 26->7, 27->6, 28->6** (alternating with the 96KB ladder; the PPU's ladder differs - 25+ still unmeasured there).

## Attempt #51: re-estimate for 8x4090/5090 at N=25-28 (solution ratio x residency step; corrects #41 which omitted steps, 2026-10-17)

Method: anchor on the 8x4090 measurements (N=24: r6=1867s/9 blocks, r7=1733s/10 blocks; N=25: r7=18,384s measured/9 blocks), extrapolate by [solution ratio x residency-step factor]; 8x5090 = 8x4090 x 0.775 (mean of the paper's 8-GPU 5090/4090 points; range 0.75-0.83). Block ladder at 96KB effective smem, WG32: rows=6 -> 9/9/8/8/8 (N=24-28); rows=7 -> 10/9/9/8/8.

| N | 8x4090 r6 | 8x4090 r7 | 8x5090 r6 | 8x5090 r7 | pick |
|---|---|---|---|---|---|
| 25 | 5.03 h | **5.11 h (measured)** | 3.90 h | 3.96 h | **6** (marginal) |
| 26 | 57.2 h | 51.6 h | 44.4 h | 40.0 h | **7 (-10%)** |
| 27 | 25.1 d | 25.5 d | 19.5 d | 19.7 d | **6** (tie, r6 edge) |
| 28 | **274 d** | 278 d | 212 d | 215 d | **6** |

**Correction to #41**: its 8x5090 column omitted the steps (r7 loses one block at N=25 and N=27, cumulative x1.25), overestimating N=27/28 by ~30% - the 8x5090 N=28 estimate goes from 5.4 months to **6.8-7.3 months** (scale band 0.75-0.83 = 208-222 days; 212 days at 0.775).

Assumptions/risks: (1) the 5090's smem/SM may differ from Ada's 100KB, shifting the ladder and possibly the per-N pick; (2) the "r6 is 3-5% faster in tie cases" edge (N=23 measured) is not included - r6 at 25/27/28 may be another 0-5% better; (3) the 27->28 solution ratio 10.9 is a trend estimate.

## Attempt #52: N=28 pick corrected (r6 on the 96KB knife-edge) + explaining the N=22 r6/r7 crossover (host-pipeline tax, 2026-10-17)

**1) N=28 pick**: r6's 8-block demand = 8x12,288 = **98,304B, exactly the 96KB convention**; measurements only prove effective smem **>=97,280B** (r7@24's 10-block demand, which indeed won by 7.2%), not >=98,304B. If the effective value is 97.3-98.3KB, r6@28 drops to 7 blocks (-12.5%); r7 needs only 94,208B (4KB margin). => **use rows=7 at N=28** (corrects #50/#51's "28->6"; 8x4090 ~278d, 8x5090 ~215d). Others unchanged: 24->7, 25->6, 26->7, 27->6.

**2) Why r7 with one extra block is still slower at N=22**: calc time is end-to-end and includes a **host pipeline that grows linearly with the subproblem count** (generation / sort+padding / const_data build / multi-GPU gather / H2D / jkl table / final weighted sum). Measured non-kernel overhead:

| platform | N=24 r7 (66.8M subs) non-kernel | rate | extrapolated N=22 r7 (29M) | N=22 r6 (3.16M) |
|---|---|---|---|---|
| 8x4090 box | 11.7s | ~0.175 s/M | ~**5.1s** | ~0.55s |
| 16 PPU | 7.7s | ~0.116 s/M | ~3.4s (measured 3.12s) | ~0.4s |

At N=22 r7 pays ~4.6s more host time than r6, while the extra resident warp saves only ~2s on a ~20s kernel -> net ~+2.6s (plus startup/noise; measured 27.2 vs 22.3). On the PPU the host pipeline is halved and the CPU is ~2x faster (gen 31M/s vs 66M/s) -> r7@22 measured 7.7% faster.

Implication: the crossover = ladder gain vs host tax; per +1 N the kernel grows x9-10 while host cost grows x1.4 -> from N=24 the host tax is <1% and the r6/r7 choice is decided by the ladder alone. The N=22 r6/r7 difference must NOT be read as "the kernel is 20% slower" - at that size the kernel difference is unresolvable (+-2-3s).

## Attempt #53: correcting #52's wording with the measured N=22 kernel/host split (2026-10-17)

New log (8-GPU box, block=64):

| run | kernel (parallel region, incl. per-GPU gather/copies) | gen | rest (sort/const_data/jkl/sum) | calc |
|---|---|---|---|---|
| r6 | **21.086 s** | 0.123 s | 1.064 s | 22.273 s |
| r7 | **21.274 s** (+0.9%) | 1.029 s | 4.876 s | 27.179 s |

**Two wording corrections (from #52 / the previous chat answer)**:
1. "r7's kernel is faster on the GPU (the +1 block saves ~2s)" - **false**. The measured kernels are essentially equal (21.09 vs 21.27s): r7's extra resident block is fully offset by its **per-subproblem fixed cost** (at N=22 trees are small and the subproblem count is x9.8, setup tax ~9%; the tax scales as 1/tree-size, i.e., /9 per +1 N - so at N=24 the measured kernel advantage r7 -7.9% = step ~10% - tax ~1%, self-consistent).
2. Attribution corrected: the 4.91s calc gap = **host side 4.72s (96.2%)** (gen +0.91s; sort/const_data/jkl/sum +3.81s) + kernel +0.19s. Nearly the entire "r7 is slower at N=22" is host-pipeline tax.
3. Rate check: r7 non-kernel = 5.905s / 29.0M = **0.204 s/M**, consistent with #52's 0.175 s/M estimate from N=24 (15% apart).

Direction unchanged (the N=22 r6/r7 gap is host tax; N>=24 decided by the ladder); wording corrected per the measurement.

## Attempt #54: 8x4090 campaign close-out review (capture-line completion, balanced/nobalanced, host-tax series, 2026-10-17)

4090 experiments ended (budget). Reviewed the new logs (both branches on 8x4090 + the logs/ balanced comparison) and the updated Excel:

**1) Capture line (faf-cuda) on 8x4090** (kernel times): r6/r7 = 1.5/1.3s (N=20), 4.1/3.6s (21), 24.9/22.2s (22), 208.2/198.4s (23), 1945.7s (24, r6; the Excel's r7@24=1882s calc has no matching run in the log). Since the capture kernel's shared memory is preset-independent, its r7 gains are NOT residency steps but finer-granularity tail/balance effects (3-11%). Reload-vs-capture kernel gap: -17% (21), -15% (22), -12% (23), -4% (24 r6) - shrinking to single digits at larger N on NVIDIA (1.8-1.9x on the PPU).

**2) balanced (mod equal split) vs nobalanced (fraction table)** (main, 8x4090, r6): mod spreads are uniformly lower (1.0/3.4/4.8/6.6% vs 5.5/12.6/19.9/16.4%), yet at N=23 mod's makespan was 5.8% worse (328.8 vs 310.9s) - single runs, inconsistent direction, needs repetition to settle. The production reload line only has mod; the PPU spread is 0.4% => no weighting needed.

**3) Host-tax series (reload r7, host-gen)**: 4.88s (22, 18% of calc), 7.62s (23, 3.9%), 11.73s (24, 0.7%), 16.59s (25, 0.09%) - negligible from N=26 on (#52/#53 closed).

**4) Finish spreads**: reload r7@24/25 = 4.6%/4.5%; r6@24 = 5.8%; capture r6@24 = 8.0%. PPU r7 = 0.4-0.5%.

**5) Excel check**: reload rows all match the logs; the capture r7@24=1882s row has no matching run in the faf-cuda log (it ends at r6@24) - confirm the source.

**8x5090 preflight (the only remaining experiment, ~1-1.5h)**: N=24 with r7/block32 (predicted ~22min; calibrates the 5090/4090 ratio 0.775 and the spread) + r6/block32 (verifies the ladder step; predicted ~28-30min); optionally block=64. Then go straight to N=26 (~40h) -> (optional N=27 ~19.7d, count known) -> N=28 (r7, ~215d ~ 7 months).

## Attempt #55: single-GPU SASS/load-balance experiments - the oracle balance ceiling (-32%) is real, but no cheap predictor exists; K constellations/thread and leaf inlining both refuted (2026-10-18)

Focused campaign on the warp lane-tail waste (space map item 1). Every count was checked against known values (N=8/10/12/20/21).

**1. Two-row inline leaf counting (do-while inner loop)**: correct but **+48%** (2925 vs 1980ms). A per-node inner loop introduces BSSY/BSYNC reconvergence points; lanes can no longer slide, and even a 15-instruction/candidate body cannot recover it.

**2. Two candidates per leaf iteration** (no inner loop, predicated): **+33%** (2640 vs 1980ms). SASS shows nvcc choosing branches instead of predication for multi-way if/else in the hot loop; loop body 33 -> 56 instructions per iteration.

**3. Moving stack pointer** (drop the 3 address instructions per iteration): first implementation produced 0 solutions - `sp -= 32` at the root pop walks out of bounds (C++ UB), and nvcc generated the inverted pointer update. A branchless `sp += 32*(descend-pop)` form gives 32 SASS/iteration (vs 33) but needs extra SEL/dual IMAD; <3%, dropped.

**4. Oracle balance ceiling measured**: added a per-thread iteration counter (`-DNQ_COUNT_ITERS`, analysis only). N=20 total iterations **727.8G**; warp max/mean of iteration counts **1.52** (the solution-count proxy reads 2.02, an overestimate). Reordering constellations inside jkl groups by the previous run's iteration counts: **1998 -> 1358ms (-32.1%)**, counts exact. The space-map item-1 bound is now measured, not estimated.

**5. Still no cheap predictor**: 2-row shallow expansion vs iterations: Spearman ~ **-0.08**; within a group, `i` (row-0 queen column) explains 12% of the within-group variance (cross-half correlation 0.996: center i expensive, edge cheap), but sorting by i only recovers 1.508 -> 1.482; sorting by (i, true cost) gives 1.205.

**6. K constellations per thread** (K=1/4/8/16, N=20): 1978 / 1972 / 2141 / 2475ms. The outer K loop **re-synchronizes lanes at every constellation boundary**, so each warp round is still the max of 32 random constellations (SIMT lockstep); K only shrinks the block count (at K=16, N=20 has only 750 blocks, less than one wave). Default K=1. This also refutes space-map item 6: a dynamic atomic queue has exactly the same per-round structure as K items; max/mean does not depend on task granularity as long as the task boundary is a warp sync point.

**7. Preset 7/8 re-measured** (same build): N=20 kernel 2265 / 2614ms vs 2011ms at preset 6, consistent with space-map item 5.

**8. Net win**: padding alignment tightened from 384 to `block_size*K` (fewer pseudo threads) + hoisting L/last/jkl loads + the K=1 restructure: N=20 **1976 -> 1914ms (-3.1%)**, N=21 **16341 -> 15843ms (-3.0%)**. Block-size table re-checked: N=20 b=128 best (1937; b=32 1998, b=64 2097); N=21 b=32 best (15923; b=64 16852).

**Conclusion**: the remaining ~30% requires a scheme that lets lanes truly slide - fine-grained tasks + stack checkpointing (preemptible DFS with task stacks in global memory) - or a high-accuracy cost predictor (shown not to exist). Static sorting, K items, and deeper presets are all out.

## Attempt #56: preemptible DFS + stack checkpoints + per-block task queue - N=20/21/22 measured -26.3%/-21.3%/-25.8% on one GPU (2026-10-18)

Implements the only remaining big lever identified in #55 (warp lane-tail waste 1.52x; oracle ceiling -32%).

**Design**
- Each block owns `block_size*K` same-jkl constellations, held in a per-block FIFO ring in global memory (`2*cap+3` ints).
- Each round: every lane atomically pops one task, restores its checkpoint (or initializes the root state), runs a `slice`-iteration budget on its private smem slab, then - if unfinished - writes {top stack levels, partial solution count, top} back to its global checkpoint and re-enqueues. Two `__syncthreads()` per round order the enqueues/accounting.
- The smem layout is unchanged (S int4 levels per lane, zero bank conflicts); queue and checkpoints live in global memory, so occupancy is untouched.

**Three findings from the implementation**
1. **The task pool shrinks linearly as tasks complete**: with cap = block_size it drains from full-lane occupancy to zero and most lanes starve late in the run (12% slower than the baseline). K=4-6 make the pool much deeper than the lane count and deliver the balance win (N=21 K sweep 4/6/8/12 -> 12661/12592/12790/13333ms).
2. **`__launch_bounds__(MAX_BLOCK, 12)` was a trap**: the 12-blocks/SM constraint capped registers at 42, so the compiler rematerialized the lane/warp base every iteration (+6 instructions, loop 33 -> 39). Occupancy is actually smem-bound (register budget ~170); dropping minBlocks gives 55 registers and a 33.5-instruction loop, N=20 1655 -> 1440ms.
3. Checkpoint/queue overhead is small: slice 4096-32768 gives 1428-1446ms, so slicing itself is not the bottleneck.

**Measured (same-session A/B; every count checked: N=6/7/8/10/12/13/14/20/21/22)**

| N | baseline (post-#55) | queue version | gain |
|---|---|---|---|
| 20 | 1938.8 ms | **1428.6 ms** | -26.3% |
| 21 | 15843 ms | **12459.9 ms** | -21.3% |
| 22 | 145957 ms | **108258 ms** | -25.8% |

The N=20 oracle bound was 1358ms (#55), so the implementation is within 5% of it. Parameters: block table unchanged (20 -> 128, 22 -> 64, 21/23+ -> 32; N=22 with b=32 is worse at 116.2s), K=6, slice=8192. Checkpoint buffers = cnt x maxd x 16B (904MB at N=22; ~4.8GB at N=26, ~11GB at N=28; divided by GPU count in multi-GPU).

**Projection**: 8x5090 N=26 ~ 40h x 0.74 ~ **29.6h**, meeting the <=30h gate. Caveat: the multi-GPU path was updated (per-GPU checkpoint/queue arrays) but cannot be tested on this single-GPU box; run an N=24 preflight on 8 GPUs first.

## Attempt #57: re-estimate N=20-28 on 8x4090 / 8x5090 with the #56 kernel (2026-10-18)

**Method**
1. Extracted each run's **GPU compute window** (8-GPU start->finish timestamps) and the host time (calc - window) from faf-cuda-reload-log/8x4090.log. The multi-GPU "CUDA kernel time" print includes per-GPU cudaMalloc/H2D/OMP startup (~1s at small N) and must not be used as the kernel time.
2. New-kernel speedup f from the single-GPU A/B: N=20 (b=128) **0.737**, N=21 (b=32) **0.787**, N=22 (b=64) **0.742**; f = **0.77** for N>=23 (production b=32, K=6). New total = old total - old GPU window x (1-f) (checkpoint allocation delta negligible).
3. N=26-28 unmeasured: anchored at N=25 and stepped by **solution ratio x ladder (blocks_prev/blocks_next)**; r6's N=25 anchor derived from its N=24 measurement, r7 uses its N=25 measurement. Residency ladder (warps/SM): r6 = 12,11,10,10,9,9,8,8,8; r7 = 12,12,11,10,10,9,9,8,8 (N=20->28).
4. 8x5090 = GPU part x **0.775** (main-branch Excel 5090/4090 ratios 0.75-0.83); host unchanged.

**Results (central estimates)**

| N | r6 4090 | r7 4090 | r6 5090 | r7 5090 | pick |
|---|---|---|---|---|---|
| 20 | 2.08 s | 3.90 s | 2.02 s | 3.81 s | r6 |
| 21 | 3.67 s | 6.99 s | 3.24 s | 6.58 s | r6 |
| 22 | 17.0 s | 21.9 s | 13.7 s | 18.6 s | r6 |
| 23 | 2.4 min | 2.5 min | 1.9 min | 2.0 min | r6 |
| 24 | 24.0 min | 22.3 min | 18.6 min | 17.3 min | r7 |
| 25 | 3.87 h | 3.93 h | 3.00 h | 3.05 h | r6 |
| 26 | 44.0 h | **39.7 h** | 34.1 h | **30.8 h** | r7 |
| 27 | 19.3 d | 19.6 d | 15.0 d | 15.2 d | r6 |
| 28 | 210 d | 213 d | 163 d | 165 d | r7 (r6 is the 96KB knife-edge) |

**Key points**
- 8x5090 N=26 r7 drops from #51's 40.0h to **~30.8h** (f in [0.73,0.79] -> 29.2-31.6h), right at the 30h gate. A preflight block-size retune (b=64 shares the pool with the same resident warps at N=23/24/28) may add 3-6%.
- For N<=21 the total is dominated by the host/per-GPU init floor (~1.7-1.8s); the new kernel only improves the total by 5-12% there. From N=22 GPU time dominates (-23%).
- r6 vs r7 at N=27/28 is inside the model error; at N=28 pick r7 because r6's 8-block demand is exactly 98,304B.
- Historical model error: r7's extra block at N=22 did not materialize (#53); the N=24 ladder step was ~80% effective; uncertainty ~±10% for N>=26.
- Preconditions: the multi-GPU path is untested on hardware; an N=24 preflight (~20 min) is needed to confirm scaling and counts.

## Attempt #58: N=23 close-out - old kernel measured 1352.4s, new kernel 1017.6s (-24.8%); b=64 refuted (2026-10-18)

Corrects #57's extrapolation: the old-kernel N=23 was **measured at 1352.4s** (not the 1314s extrapolation).

**New data**
| N=23 r6 | time | note |
|---|---|---|
| old kernel (existing measurement) | 1352.4 s | baseline |
| new kernel b=32 (default) | **1017.6 s** | **-24.8%**, f = 0.752 |
| new kernel b=64 | 1195.5 s | +17.5%, count correct but slower |

- f measurements: N=20 **0.737**, N=21 **0.787**, N=22 **0.742**, N=23 **0.752** (mean ~0.755); #57's 0.77 was slightly conservative.
- **b=64 is 17.5% slower at N=23**: its per-block dynamic smem is 19,456B, and with the static jkl table and allocation granularity the resident block count drops (b=32 keeps 10 warps). "Same warps -> bigger block" does not hold; the table stays 23-28 -> 32. b=64 won at N=22 because the per-block smem there (9,216B) still allows 10 warps.
- **Ladder warning**: the N=22->23 step is 9.71x (new), 9.27x (old), vs the pure solution ratio 9.005x. The new kernel loses 4.8% more than the old one on this step, so the N>=26 ladder extrapolations still carry ~±10% uncertainty.

**Updated estimates with f=0.752**: 8x4090 N=23 ~ 182.4 x 0.752 + 2.3 ~ **139.5 s** (single-GPU cross-check 1017.6/7.20 ~ 141 s); 8x5090 N=23 ~ **1.8 min**; 8x5090 N=26 r7 ~ **30.1 h** (was 30.8h), N=27 r6 ~ 14.6 d, N=28 r7 ~ **162 d**.

## Attempt #59: ALU/latency work toward the 25h target - another ~10-11% single-GPU (N=20/21/22 interleaved, 2026-10-18)

Goal: another 17% on top of #56 (8x5090 N=26 30.1h -> 25h), benchmarked at N=21.

**Diagnosis**: the hot loop was 34 instructions/iteration with ~14-17 ALU (half-rate) ops; at N=21 each warp-iteration takes ~154 cycles (30.5 instructions -> only ~55% issue utilisation, ~52% ALU-pipe busy) - **neither issue- nor ALU-saturated: it is LDS-dependency latency with too few warps**. Occupancy is smem-bound (11 warps), and the "one level less stack" A/B was neutral (12 blocks do not fit), so adding warps alone is not available; only fewer instructions / shorter chains.

**Four changes (all counts verified, compute-sanitizer clean)**
1. **True stack depth + guard slot**: the leaf level never pushes, so host maxd becomes `(N-2)-start` (the old +1 was conservative). Each warp keeps one extra "guard" level below its stack, written per task as `{col, last, last, 0}`: if the second unrolled half runs after a root pop it reads the guard - residual=0 (no push), left|right=last (child=0), pc=popc(col)=init_pc (jkl row = start, always in bounds). This **removes the mid-loop top check** (-1 ALU, -1 branch per iteration).
2. **Right shift on the FMA pipe**: `(right|p)>>1` -> `__umulhi(right|p, 0x80000000u)` (IMAD.HI); the last SHF disappears from the loop.
3. **Child in 2 LOP3**: inline `lop3.b32` forces `t=cur|left|right`, `child=last&~(t|jkl)`, avoiding the compiler's third LOP3 for the child!=0 test (the test folds into the LOP3 predicate output).
4. **Merged loop condition**: `while (top>0 && budget>0)` with the budget decrement at the loop tail; one instruction fewer per pair.

**Interleaved A/B (same thermal state, all counts exact)**

| N | before | after | gain |
|---|---|---|---|
| 20 | 1506 ms | **1356 ms** | -10.0% |
| 21 | 13193 ms | **11730 ms** | -11.1% |
| 22 | 109186 ms | **102730 ms** | -5.9% (last two changes only; ~-10% vs the first build) |

Hot loop 63 -> 61 SASS per pair (30.5/iteration). K=6, slice 8k-16k flat (keep 8192). Fixed the multi-GPU smem formula to `(maxd+1)` levels (otherwise the guard is out of bounds).

**Status**: 8x5090 N=26 estimate 30.1h x ~0.90 ~ **27.1h**, still ~8% above the 25h target. The remaining bottleneck is smem-limited occupancy (LDS dependency chain at 11 warps). Next candidate: spill the rarely-visited shallow stack levels (near the root) to global memory and keep only the deep D levels in smem (the leaf level is the most visited) - this would cut the per-warp slab from 21 to ~12-14 levels (+30-50% warps) at the cost of one boundary check per iteration and a few global accesses. Structural change; needs its own verification.

## Attempt #60: corrected real-occupancy ladder + shallow-stack spill refuted (2026-10-18)

Still after 25h for 8x5090 N=26 (from ~27.1h). This round tested whether more warps are available and what they cost.

**1. Real occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor, N=21 b=32)**
Each block carries ~**1KB fixed overhead** and each SM has ~**100KB usable** (not the old 96KB-without-overhead convention):
`warps/SM = floor(100352 / (levels*512 + 1024))`, levels = N-4 (r6) / N-5 (r7, guard included).

| N | r6 old | r6 real | r7 old | r7 real |
|---|---|---|---|---|
| 20 | 12 | **10** | 12 | **11** |
| 21 | 11 | **10** | 12 | **10** |
| 22 | 10 | **9** | 11 | **10** |
| 23 | 10 | **9** | 10 | **9** |
| 24 | 9 | **8** | 10 | **9** |
| 25 | 9 | **8** | 9 | **8** |
| 26 | 8 | **8** | 9 | **8** |
| 27 | 8 | **7** | 8 | **8** |
| 28 | 8 | **7** | 8 | **7** |

Measured N=21 occupancy vs smem levels: 17->10 warps, 15->11, 14->12, 13->13, 12->14, 11->15, 10->16, 8->19, 6->24.
**Impact on estimates**: the ladder shifts down by 1-2 warps, but the cumulative N=25->28 step is nearly unchanged (r7 1.125->1.143, r6 same ~1.14), so the #57 N=26-28 estimates (~27.1h / 162d) still hold within ~2%. The old table must be updated as above.

**2. What one warp is worth** (measured by over-allocating smem to force lower occupancy; counts correct): at N=21, 10->9/8/7 warps = 11685/12616/13733/15290ms, i.e. **+8.0%/+17.5%/+30.8%** - more warps really help, with a steeper slope at lower occupancy.

**3. Shallow-stack spill: implementation refuted.** Design: keep only the deep D levels in smem, spill the shallow S-D levels to the existing global checkpoint array, and use a per-lane mode base so hot-loop addressing stays a single add; B-level copies only at boundary crossings (plus guard/checkpoint/restore mode handling). Measured:
- **B=0 (no spill, code merely present)**: N=20 2.97s vs the 1.35s baseline (**2.2x slower**);
- spill=12 levels 5.25s, 10 levels 13.3s, 8 levels 19.0s (up to 14x slower).
Cause: the crossing copy loops plus the mode checks make the hot loop branch/BSSY and raise register pressure; the occupancy gain (10->14 warps, ~+20-25% theoretical) cannot pay for it. Reverted (source restored to the #59 commit; counts re-verified).

**4. Depth-class kernel split refuted**: the N=20 start distribution is 3/4/5 = 68.6%/23.4%/8.0% of constellations, but weighted by b^(N-2-start) the work is **86.6%/11.8%/1.6%** - the deep-stack class is the bulk, so splitting by depth buys nothing.

**Conclusion**: occupancy remains the most valuable lever (+20-30% potential), but it needs a **zero-check hot loop** (e.g. warp-uniform modes, or a scheduling-level scheme that leaves the loop untouched); the mixed-addressing/mode-switch design inside this single flat loop is refuted. The #59 build (4d1f129) is the final state; 8x5090 N=26 ~ 27.1h.

## Attempt #61: register-carried state machine refuted (2026-10-18)

**Motivation**: leaf-level iterations are over half of all iterations, and consecutive candidates of the same node ("stay") reload the unchanged state (LDS.128) and store the residual (STS.32) every trip. Keeping the current node state in registers, saving the parent only on descent and reloading it on pop, would cut ~2 LDS + 2 STS per iteration to about one memory op on average.

**Design**: state in registers; on descent store the parent state into the parent slot (when the parent is exhausted, the child state overwrites the parent slot = collapse); on pop reload the new stack top; leaf stays touch no smem; the slice checkpoint first writes the register state into the top slot, then copies the stack.

**Process**: the first version omitted the collapse (an exhausted parent's slot was not overwritten) -> popping back to a residual=0 parent with p=0 drifted the state and hung at N=8. With the collapse added, counts are exact at N=8/10/12/13/14.

**Measured (N=20, interleaved A/B)**: **1913ms vs 1357ms (+41%)**. Same mechanism as before: conditional LDS/STS inside the hot loop make nvcc emit branches/reconvergence (#33/#55/#60), and the long-lived state registers hurt scheduling.

**Conclusion**: on-demand loads / conditional memory access variants inside the single flat loop are all off the table; **unconditional accesses + reload** is the form the compiler schedules best. Reverted to the #59 commit (N=20/21 counts and times re-verified). The final build remains `4d1f129`; 8x5090 N=26 ~ 27.1h.

## Attempt #62: 10/19 re-test (8x4090 / 16 PPU) + r6/r7 decision and updated estimates (2026-10-19)

**Data** (reload_8_4090.txt / reload_16_ppu.txt; per-card median durations; 8-card spread 1-7%):

| N | r6 | r7 | r8 | r7 vs old kernel |
|---|---|---|---|---|
| 20 | 0.5 s | 0.3 s | - | -35% |
| 21 | 3.0 s | 1.5 s | - | -32% |
| 22 | 18.7 s | 12.2 s | 12.4 s | -39% |
| 23 | 145.0 s | 115.5 s | 107.2 s | -36% |
| 24 | 1398.8 s | 1092.6 s | - | -35% |
| 25 | - | 11927.4 s | - | -35% |

- r7's GPU window / old kernel = 0.61-0.68, matching the cumulative prediction **0.666** (#56's 0.74 x #59's 0.90); N=25 total **11,948 s** vs the predicted 12,251 s (-2.5%).
- **The r6 regression is real and explainable** (not a measurement error; per-card spreads are normal): r6 has 8-10x fewer constellations with 8-10x larger tasks, which amplifies the queue's pool-drain tail (the last few big tasks run nearly serially); r7/r8 have small tasks and stay at a uniform -35%.
- **r8**: kernel another ~7% faster (N=23 107.2 vs 115.5 s) but the host pipeline/memory explode (N=23 host 79 s, total 187.8 s; N=26 would be ~1.3B constellations ~ 42 GB host + ~21 GB GPU constant array) -> not viable.

**Decisions**
- **Preset: r6 for N=20/21 (host-bound), r7 for N>=22.** The old "switch to r7 at N=24 / use r6 at 25 and 27" is obsolete; r6 is 16% behind at N=22 and 20-22% behind at N=23/24.
- New anchor (8x5090, f=0.775): **N=26 r7 ~ 26.0 h** (8x4090 33.5 h), N=27 r7 ~ 11.4 d (14.7 d), N=28 r7 ~ **142 d / 4.7 months** (183 d). Only ~4% above the 25 h target.
- **Memory constraint**: N=26 checkpoint buffers ~ 150M x 21 x 16B / 8 ~ **6.3 GB per GPU** (8 cards OK); **single-GPU N>=26 needs ~50 GB, beyond a 24 GB card - single-GPU large N is infeasible**. Host-side constellation arrays at N=26 are ~5-7 GB.

**"Not yet tried" re-review**: ILP with 2 constellations (no: two stacks halve the warps, each warp is worth 6-10%); global queue (no: needs register-computed jkl, block-level scheduling already balances, <=2% upside); preset (settled: r7 for N>=22); algorithmic (forward checking is already exact in the bitmask, no compact table for the last rows, all three leaf-batching implementations measured slower) -> low expected value.

## Attempt #63: scheduling-level tuning (r7: K=16 + slice=16384) - N=22 r7 -3.9% (2026-10-19)

Goal: 8x5090 N=26 from ~26.0h to 24h. Only scheduling was changed (hot loop untouched).

**Sweeps (single GPU, N=22 r7; equal warps; all counts exact)**
- K: 4->98.5s, 6->95.8s, 8->95.1s, 12->94.4s, **16->93.4s**, 24->93.8s, 32->93.9s.
- At K=16: b=64 93.2s (~= b=32 93.4s); slice 4096->94.2, 8192->93.4, **16384->93.0**, 32768->92.8, 65536->93.6s.
- Mechanism: r7 tasks are 8-10x smaller than r6, so a deeper pool shortens the queue's pool-drain tail; beyond K~16 the larger alignment/block-count effects win.
- **New defaults: K=16 for preset>=7, K=6 for r6; slice=16384** (NQ_K/NQ_SLICE still override).
- Measured with the new defaults: N=22 r7 **92.06s (vs 95.8s at K=6/8192, -3.9%)**; N=20 r6 1307ms, N=21 r6 11110ms (counts exact, slightly faster too).

**Impact on 8x5090 N=26**
- 26.0h x 0.96 ~ **25.0h** (at the 5090/4090 ratio 0.775); if the large-N kernel ratio is close to the Excel's N=23 value **0.754**, then ~**24.4h**; 0.74 -> 23.9h.
- So the **24h target is at the edge of the error bar** and hinges on the real 5090 ratio - it must be measured with an 8x5090 N=24 r7 preflight using the new defaults (expect 8x4090 1128s x ~0.75 ~ 14 min).
- If ~1-2% is still missing after the preflight, the only remaining option is task splitting during the pool drain (extra checkpoint slots + a task->original-gid map; ~+1.5-2%, sizable change).

## Attempt #64: leaf-row jkl preload - refuted (2026-10-19)

**Motivation**: at `pc==leaf_pc` the jkl row is fixed, so it can be preloaded per task; the leaf level is the largest iteration population, so this could remove the jkl LDS dependency latency there. Implementation: preload `jrow_leaf = jkl[jrow_base+leaf_pc]`, then in the loop `at_leaf = (pc==leaf_pc)`, `jrow = at_leaf ? jrow_leaf : jkl[...]`, reusing `at_leaf` for the leaf test.

**Result**: counts exact (N=8/10/12/13/14), but the N=20 interleaved A/B gives **1388ms vs 1289ms (+8%)**. nvcc still adds select/scheduling overhead for the conditional LDS, cancelling the latency saving.

**Conclusion**: same mechanism as #55/#60/#61 - any conditional memory access inside the hot loop is off the table. Reverted (source stays at 9c0970b). The single-GPU kernel is at its plateau: the only measurable losses left are the queue drain tail (~1.5% at K=16) and the multi-GPU per-card imbalance (~1.7% at N=25), both scheduling-level rewrites (task splitting / weighted card split) worth ~1.5% each.

## Attempt #65: N=25 r6 filled in (8x4090 = 13,332s) + guard-removal variant refuted (2026-10-19)

**1. N=25 r6 (the last missing cell)**
- 8x4090 N=25 r6 = **13,332 s** (3.70h), count exact.
- Same box N=25 r7 = 11,948 s (K=6/8192; the new K=16/16384 default is another ~4% faster) -> **r6 is 11.6% slower**.
- Conclusion: N=25 also uses **r7** (consistent with #62's mechanism: r6's 8-10x larger tasks amplify the queue tail; the old "r6 at N=25" pick is obsolete). r6 remains only for N=20/21 (host-bound).

**2. Guard-removal variant (aimed at the 5090's 128KB) - refuted**
- Motivation: at N=26 r7 the stack is 20 levels + 1 guard level = 21; on the 5090's 128KB/SM this gives 10 warps, while 20 levels (no guard + mid-loop check) would give 11 (+10%).
- Reproduced the same trade-off on the 4090 at N=22 r6 (model: 9 vs 10 warps), same-conditions A/B: **no-guard 101,890 ms vs guard 97,645 ms (+4.3%)**, counts exact.
- The mid-loop `top>0` check costs more than the modeled warp gain, and the 4090 measurement suggests that warp delta may not materialize. **Keep the guard** (source stays at 9c0970b); do not switch on the 5090 unless the preflight shows a real one-warp drop.

## Measured Optimization Space Map (2026-10)

An instrumented campaign (per-thread iteration counters, warp-level analysis, occupancy queries, SASS dumps) bounded what remains after #12:

1. **Warp lane-tail waste = 1.52x — the biggest single loss (captured by #56).** A warp runs until its slowest lane finishes; measured warp-max/warp-mean of per-subproblem iteration counts: median 1.49 (#55 re-check: 1.52). #55 measured the bound with an iteration-count oracle reorder: 1998 -> 1358ms (-32.1%); #56 implemented it with preemptible DFS + stack checkpoints + a per-block task queue: 1938.8 -> 1428.6ms (-26.3% at N=20), within 5% of the oracle, now the production kernel.
2. **No cheap size oracle exists.** Rank correlation with true per-subproblem iterations: start row -0.35; `popc(valid at start)` 0.14; 1-row expansion 0.14; 2-row 0.14; deeper frontier counts (n3..n5) decay to zero and go *negative* (wide shallow fans die early). Sorting by the best combination recovers only 2.6% of the waste (1.521 → 1.481). Dead roots at N=20: none.
3. **Utilization**: reload (10 warps/SM, 35 SASS/iter) ≈ 1.1 IPC/SM; capture kernel (32 warps/SM, 48 SASS/iter) ≈ 1.5 IPC/SM — i.e. 28–37% of the 4 IPC peak. Tripling resident warps buys only ~1.3x IPC.
4. **Register-resident loop variants** (state in registers, smem touched only on push/pop): two-pass (pop as its own iteration) 3.3x slower (warp divergence serializes both paths); single-pass with conditional save + popc-based row recompute 1.6x slower. The compiler's scheduling of the simple unconditional-reload loop wins.
5. **Deeper presets**: preset 7: 2358ms, preset 8: 2667ms (vs 2172ms at 6) — smaller subtrees have relatively larger variance, so the warp-tail waste grows.
6. **Dynamic task fetching (coarse version refuted by #55; fine version implemented by #56)**: coarse tasks (one whole constellation) do not help - the 1.52 factor is independent of task size (K=1/4/8/16 measured 1978/1972/2141/2475ms). #56's fine-grained version (task = one slice of a constellation, with stack checkpointing) lets lanes truly slide at budget boundaries, and fixes the pool-shrink starvation (K=4-6 keep the pool deep).

## Approaches Not Yet Tried (reviewed 2026-10-19, #62)

1. **ILP with 2 constellations per thread**: not worth it - two independent searches need two stacks (2x smem per lane), halving resident warps; each warp is worth 6-10% (#60), so the ILP gain cannot compensate.
2. **Persistent threads with a global work queue**: not worth it - giving up the per-block jkl table requires register-computed constraints (+5-7 ALU/iteration on a latency/ALU-sensitive loop); block-level scheduling already balances SMs and the per-card spread is 1-7%.
3. **Different preset_queens**: settled by #62 - r6 for N=20/21 (host-bound) and r7 for N>=22 (r7 is 16-35% faster at N=22-25 because smaller tasks shorten the queue's drain tail). r8's kernel is ~7% faster but its host/memory explode.
4. **Algorithmic changes**: low expected value - the bitmask DFS already applies exact forward checking; table-driven last-row counting has no compact index; all three leaf-batching implementations (inline two rows, two candidates per trip, register-carried state) measured 33-48% slower due to SIMT divergence / conditional memory access.
