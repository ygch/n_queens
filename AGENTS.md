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
- **`src/n_queens_cuda.cu`**: faf's D4 constellation tree (4 border queens i,j,k,l + preset queens + per-workgroup `jkl_queens` table) × main's reload state machine — 16B/level `int4` full-state stack `{cur,left,right,residual}` (slot `e*32+l`, zero bank conflict), per-trip reload, **no** capture bits / anti-revisit mask; leaf counting via `__popc(child)` at row N-2 (second-to-last-row optimization); manual unroll×2; exact-depth dynamic shared memory.
- All N=6–28 dispatch to the reload kernel via a `switch` in `cuda_solve()`; block size comes from a **per-N table** (20→128, 21→32, 23→32, 24–28→32, else 64; `NQ_BLOCK` env var overrides for sweeps). Multi-GPU: blocks are assigned to GPUs by `blk % gpu_num` (mod-interleaved, `solve_multi_gpu()`).
- Elsewhere: the original **capture** kernel (4B/level stack + `ld_mem`/`rd_mem` + funnelshift + `direction` trick) is frozen on branch `faf-cuda`; the int4+PTX legacy solver lives on `main` (`gpu_solver.cpp`).

`constellation.cpp` implements D4 symmetry enumeration (rotation/mirror elimination, `symmetry()` returns multiplier 8/4/2). Kernels are **template-instantiated per N** (6–28).

## Critical Performance Knowledge

- The kernel family is **throughput-bound (IPC-limited)**, not latency-bound. RTX 4090 has 4 warp schedulers per SM; demand far exceeds supply. Increasing occupancy does NOT help by itself.
- The reload kernel's per-trip full-state reload **is** the optimal form — register-carried / delta-stack variants were refuted five times (warp-divergence tax; the compiler schedules the plain unconditional-reload loop better).
- Reload vs capture (same-session A/B): −12…−13% at N=19–21, −8.1% at N=22 (149.1s vs 162.2s), −8.0% at N=23 (1352.4s vs 1469.7s). κ model: reload ≈ 3.6–4.0× per-warp throughput; flip points are ruled by per-block smem → resident blocks → resident warps.
- The optimal `preset_queens` follows the maxd residency ladder (each +1 preset row shrinks maxd by 1 → 512B less per warp): paper single GPU 5; 8×4090/5090: 6 for N≤23, **7 for N≥24** (measured at N=24 on 8×4090); 16 PPU: 7 for N≥22. Note N=6/7 require preset ≤ N−2.
- See `OPTIMIZATION_LOG.md` and `优化日志.md` for the full attempt record (#1–#34).

## Branch Strategy

- `main`: fastest original solver (int4 + PTX, ~3.6s for N=20)
- `faf-opencl`: Step 1 — minimal C/C++ port of NQueensFAF's OpenCL code
- `faf-cuda`: Step 2 — original faf **capture** CUDA port (frozen reference implementation)
- `faf-cuda-reload` (**this branch**): reload-only production line — the 2026-10 optimization campaign (per-N block-size table, exact-depth stack, unroll×2)
- `v4-mid-block` and others: experiment archive (`v4-mid-block` carries the full 39-commit campaign incl. refuted ideas; `faf-reload-loop` is a redundant ancestor of it)

## Gotchas

- Stack layout: lane `l`'s level `e` sits at `sbase + e*32` with `sbase = (lid>>5)*(S*32) + (lid&31)` — never break the stride-32 (bank conflicts).
- N=24–28 production config (reload @ block size 32) was κ-model projected; block=32 is now **validated at N=24 on 8×4090** (beat block=64 by 5.4%, rows=6). N=25–28 still projected.
- Capture-kernel traps — `__funnelshift_l(lo, hi, s)` argument order; anti-revisit mask `free &= ~(queen + direction - 1)` — apply only on `faf-cuda`.
- NQueensFAF reference project is at `../NQueensFAF/` (OpenCL baseline). `tools/nqueens_replica.py` is the per-frame Python oracle used to validate kernel semantics.
