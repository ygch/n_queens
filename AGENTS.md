# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Build & Run

```bash
cd src && bash compile.sh        # builds ./src/n_queens
./src/n_queens <N> [preset_queens]   # N in [6,28], preset_queens default 6
```

Compile target is **sm_89** (RTX 4090). Requires `nvcc` with C++17 and `-allow-unsupported-compiler` (needed for newer gcc/nvcc combinations). `--ptxas-options=-v` outputs register/smem stats — use these to verify optimization changes.

## Architecture

This branch (`faf-cuda-reload`) has a **single production kernel — the reload form**:
- **`src/n_queens_cuda.cu`**: faf's D4 constellation tree (4 border queens i,j,k,l + preset queens + per-block `jkl_queens` table) × main's reload state machine — 16B/level `int4` full-state stack `{cur,left,right,residual}` (slot `e*32+l`, zero bank conflict), per-trip reload, **no** capture bits / anti-revisit mask; leaf counting via `__popc(child)` at row N-2 (second-to-last-row optimization); manual unroll×2; exact-depth dynamic shared memory.
- **Preemptible scheduling (#56, production)**: each block owns `block_size*K` same-jkl constellations held in a per-block global-memory FIFO queue; every round each lane pops a task, runs a `slice`-iteration budget, then checkpoints `{stack, partial solutions, top}` to global memory and re-enqueues (two `__syncthreads()` per round). The smem layout/occupancy are unchanged. Kernel args: `K`, `slice`, `ckpt/ckpt_top/ckpt_sol/queue`. Defaults: **K=16 for preset≥7 (K=6 for r6), slice=16384** (`NQ_K`, `NQ_SLICE` override; #63).
- All N=6–28 dispatch to the reload kernel via a `switch` in `cuda_solve()`; block size comes from a **per-N table** (20→128, 21→32, 23→32, 24–28→32, else 64; `NQ_BLOCK` env var overrides for sweeps). Each block covers `block_size*K` constellations of one jkl group (`K` = queue pool depth per lane; default **16 for preset≥7, 6 for r6**, #63; padding aligns to `block_size*K`). Multi-GPU: blocks are assigned to GPUs by `blk % gpu_num` (mod-interleaved, `solve_multi_gpu()`).
- Elsewhere: the original **capture** kernel (4B/level stack + `ld_mem`/`rd_mem` + funnelshift + `direction` trick) is frozen on branch `faf-cuda`; the int4+PTX legacy solver lives on `main` (`gpu_solver.cpp`).

`constellation.cpp` implements D4 symmetry enumeration (rotation/mirror elimination, `symmetry()` returns multiplier 8/4/2). Kernels are **template-instantiated per N** (6–28).

## Critical Performance Knowledge

- The kernel family is **latency/pipe-bound at the production occupancies**: measured at N=21, 10→9/8/7 warps costs +8.0%/+17.5%/+30.8% (#60), i.e. each resident warp is worth ~6–10% until the ALU pipe saturates. Occupancy is smem-limited (see the real 1KB/block-overhead ladder below), so more warps require less smem per warp — but any hot-loop mode switching to achieve that is fatal (#60).
- The reload kernel's per-trip full-state reload **is** the optimal form — register-carried / delta-stack variants were refuted five times (warp-divergence tax; the compiler schedules the plain unconditional-reload loop better).
- Reload vs capture (same-session A/B): −12…−13% at N=19–21, −8.1% at N=22 (149.1s vs 162.2s), −8.0% at N=23 (1352.4s vs 1469.7s). κ model: reload ≈ 3.6–4.0× per-warp throughput; flip points are ruled by per-block smem → resident blocks → resident warps.
- **Preset selection with the preemptible kernel (#62): r6 for N=20/21 (host-bound) and r7 for N≥22.** On 8×4090 r7 is 16–35% faster than r6 at N=22–25 because r6's 8–10× larger tasks amplify the queue's pool-drain tail; the old residency-ladder pick (“6 at N=25/27, 7 at N=24/26/28”) is obsolete. r8's kernel is another ~7% faster but its host pipeline/memory explode (~1.3B constellations at N=26 → ~42GB host + ~21GB GPU constant array). Real occupancy for sizing: `warps/SM = floor(100352 / (levels*512 + 1024))` with ~1KB fixed per-block overhead (levels = N−4 for r6 / N−5 for r7, guard included), #60. N=6/7 require preset ≤ N−2.
- **Warp lane-tail waste (1.52× at N=20) was the biggest single-GPU loss and is now captured.** A warp runs until its slowest lane finishes. Oracle reordering by true iteration counts gives −32.1% (#55); the production preemptible scheduler (#56) + ALU work (#59) gives **N=20 1938.8→1428.6ms (−26.3%), N=21 15843→12459.9ms (−21.3%), N=22 145957→108258ms (−25.8%), N=23 1352.4s→981.8s (−27.4%)**, all counts exact; on 8×4090 r7 the combined gain is a uniform −32…−39% at N=20–25 (N=25: 18,384→11,948s), which puts the 8×5090 N=26 r7 estimate at **~26h** (#62) (f≈0.74–0.79; do not assume larger blocks win on equal warps — b=64 at N=23 was 17.5% slower due to the smem allocation granularity) — within ~5% of the oracle. Coarse dynamic fetch / K-per-thread without preemption does not help (SIMT lockstep at task boundaries); the fine-grained slice+checkpoint version does.
- Occupancy is the remaining lever (~+8%/warp near 10 warps, #60), but any hot-loop mode switching is fatal: a shallow-stack spill (mode base + rare boundary copies) was 2.2x slower even with the spill disabled, and up to 14x with it enabled. Depth-class kernel splits do not help (start=3 constellations carry 87% of the work).
- Analysis tools: `NQ_DUMP=<path>` dumps per-constellation solutions (`.bin`), ijkl (`.bin.ijkl`) and iteration counts (`.bin.iters`, needs `-DNQ_COUNT_ITERS` in the kernel).
- See `OPTIMIZATION_LOG.md` and `优化日志.md` for the full attempt record (#1–#62).

## Branch Strategy

- `main`: fastest original solver (int4 + PTX, ~3.6s for N=20)
- `faf-opencl`: Step 1 — minimal C/C++ port of NQueensFAF's OpenCL code
- `faf-cuda`: Step 2 — original faf **capture** CUDA port (frozen reference implementation)
- `faf-cuda-reload` (**this branch**): reload-only production line — the 2026-10 optimization campaign (per-N block-size table, exact-depth stack, unroll×2)
- `v4-mid-block` and others: experiment archive (`v4-mid-block` carries the full 39-commit campaign incl. refuted ideas; `faf-reload-loop` is a redundant ancestor of it)

## Gotchas

- Stack layout: lane `l`'s level `e` sits at `sbase + e*32` with `sbase = (lid>>5)*(S*32) + (lid&31)` — never break the stride-32 (bank conflicts).
- Hot-loop codegen: any inner loop or multi-way if/else in the DFS body makes nvcc emit branches + BSSY/BSYNC instead of predication (33→56 SASS/iteration) and costs +33–48% even when it does less work (#55). Keep the body flat and single-path.
- Do **not** re-add `__launch_bounds__(MAX_BLOCK, minBlocks)` with a high minBlocks: 12 blocks/SM caps registers at 42 and forces per-iteration rematerialization of the lane/warp base (loop 33→39 SASS, +16%). Occupancy here is smem-bound; without minBlocks the kernel uses ~55 regs and 33.5 SASS/iteration (#56).
- Queue protocol invariants: all lanes of a block must execute the same number of `__syncthreads()` (the loop exits uniformly when `qtail-qhead==0`); the pool depth is `block_size*K` and must stay ≫ lanes or late-run lanes starve (K=4–6 measured). Checkpoint memory is `cnt*maxd*16B`: at N=26 r7 (~150M constellations) that is **~6.3GB per GPU on 8 cards but ~50GB single-GPU — single-GPU N≥26 does not fit a 24GB card**; host-side constellation arrays are ~5–7GB at N=26.
- N=24–28 production config (reload @ block size 32) was κ-model projected; block=32 is **validated at N=24 on 8×4090**, and with the queue kernel **r7 beats r6 by 21–22% at N=23/24** (#62). N=25–28 still projected.
- Capture-kernel traps — `__funnelshift_l(lo, hi, s)` argument order; anti-revisit mask `free &= ~(queen + direction - 1)` — apply only on `faf-cuda`.
- NQueensFAF reference project is at `../NQueensFAF/` (OpenCL baseline). `tools/nqueens_replica.py` is the per-frame Python oracle used to validate kernel semantics.
