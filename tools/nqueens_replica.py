#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nqueens_replica.py — faf 星座法内核的 Python 精确复刻器（调试 / 验证工具）

用途
----
修改 CUDA D4 星座内核（src/n_queens_cuda.cu）后，先在真实子问题帧上用本工具
验证新算法的"逐子问题计数"是否与参考实现完全一致，再上 GPU 实测。可把调试
代价从"每次编译 + 跑 N=20"降到秒级。开发回载式内核期间，此法在写 CUDA 前
就抓出了死根守卫缺失与计数语义偏差两类问题。

支持的复刻版本（两者在全部测试帧上逐子问题计数完全一致，1747 帧 0 误差）
--------------------------------------------------------------------------
orig   原始捕获式循环：4B/层 queen 栈 + ld_mem/rd_mem funnelshift 反向恢复，
       对应 n_queens_kernel_cap（N>=22 路径，即尝试 #12 之前的旧内核）。
reload 回载式循环：16B/层 int4 栈存完整状态 {cur,left,right,residual}，每轮
       重载父状态；叶判据 popc(cur)==popc(col)+(N-start-2)，见优化日志 #12，
       对应 n_queens_kernel（N<=21 路径）。

如何生成帧文件
--------------
在 src/n_queens_cuda.cu 的 cuda_solve() 中、padWithPseudo() 之后插入：

    {   // DEBUG FRAME DUMP
        FILE *fd = fopen("/tmp/frames.txt", "w");
        for (long long i = 0; i < (long long)constellations.size(); i++) {
            int s = constellations[i].start_ijkl >> 20;
            if (s == 69) continue;                 // 跳过伪星座
            int ijkl = constellations[i].start_ijkl & 0xFFFFF;
            fprintf(fd, "FRAME ld=%u rd=%u col=%u start=%d j=%d k=%d l=%d\\n",
                    (unsigned)constellations[i].ld, (unsigned)constellations[i].rd,
                    (unsigned)constellations[i].col, s,
                    (ijkl >> 10) & 31, (ijkl >> 5) & 31, ijkl & 31);
        }
        fclose(fd);
    }

重新编译后运行 ./n_queens <N> [preset]，再用本工具分析生成的帧文件。

用法
----
    python3 nqueens_replica.py frames.txt               # 对比 orig 与 reload，打印不一致帧
    python3 nqueens_replica.py frames.txt --sim orig    # 只跑一种版本，输出逐帧计数（CNT <idx> <count>）
    python3 nqueens_replica.py frames.txt --sum         # 只打印两个版本的计数总和

提示：验证新内核实现时，可让内核把每个子问题的结果也按 gid 顺序打印，
与 `--sim orig`（或 reload）的 `CNT <idx> <count>` 输出逐行 diff。
注意：逐子问题计数为"原始解数"（不含 symmetry 权重），与内核 results[] 一致。
"""

import re
import sys

M32 = 0xFFFFFFFF


def popcount(x):
    return bin(x & M32).count('1')


def build_jkl(N, j, k, l, L):
    """jkl_queens 表：与 host 侧构造完全一致（边界皇后 j/k/l 的逐行对角线约束）。"""
    ldiagbot = (L >> j) | (L >> l)
    rdiagbot = (L >> j) | (L >> (N - 1 - k))
    ldiagtop = L >> k
    rdiagtop = 1 << l
    jkl = [0] * N
    for a in range(N):
        if a == k:
            jkl[a] = (~L) & M32
        elif a == l:
            jkl[a] = (~1) & M32
        else:
            jkl[a] = ((ldiagbot >> (N - 1 - a)) | (rdiagbot << (N - 1 - a))
                      | (ldiagtop << a) | (rdiagtop >> a) | L | 1) & M32
    return jkl


def _setup(N, c_ld, c_rd, c_col, start, j, k, l):
    L = 1 << (N - 1)
    jkl = build_jkl(N, j, k, l, L)
    ld = c_ld
    rd = c_rd
    col0 = (~(L - 2) & M32) ^ c_col
    if k >= start:
        ld &= ~(L >> (k - start)) & M32
    if l != N - 1 and l >= start:
        rd &= ~(1 << (l - start)) & M32
    last = (1 << N) - 1
    return L, jkl, ld, rd, col0, last


def sim_orig(N, c_ld, c_rd, c_col, start, j, k, l, maxsteps=5_000_000):
    """原始捕获式循环（mem 版）：queen 栈 + funnelshift 捕获位恢复。"""
    L, jkl, ld, rd, col0, last = _setup(N, c_ld, c_rd, c_col, start, j, k, l)
    ld_mem = 0
    rd_mem = 0
    row = start
    solutions = 0
    free = (~(ld | rd | col0 | jkl[row])) & M32
    queen = (-free & M32) & free
    queens = {start: queen}
    direction = 0
    steps = 0
    while row >= start:
        steps += 1
        if steps > maxsteps:
            return None
        if free:
            if row == N - 2:
                solutions += 1
                direction = 0
                row -= 1
                queen = queens.get(row, 0)
                ld = (((ld >> 1) | (ld_mem << 31)) & M32) & (~queen & M32)
                rd = (((rd << 1) | (rd_mem >> 31)) & M32) & (~queen & M32)
                ld_mem >>= 1
                rd_mem = (rd_mem << 1) & M32
            else:
                direction = 1
                queen = (-free & M32) & free
                queens[row] = queen
                row += 1
                ld_mem = ((ld_mem << 1) | (ld >> 31)) & M32
                rd_mem = ((rd_mem >> 1) | ((rd & 1) << 31)) & M32
                ld = ((ld | queen) << 1) & M32
                rd = ((rd | queen) >> 1) & M32
        else:
            direction = 0
            row -= 1
            queen = queens.get(row, 0)
            ld = (((ld >> 1) | (ld_mem << 31)) & M32) & (~queen & M32)
            rd = (((rd << 1) | (rd_mem >> 31)) & M32) & (~queen & M32)
            ld_mem >>= 1
            rd_mem = (rd_mem << 1) & M32
        if row < start:
            break
        free = (~(jkl[row] | ld | rd | col0)) & M32
        free = free & (~((queen + direction - 1) & M32) & M32)
        col0 = (col0 ^ queen) & M32
    return solutions


def sim_reload(N, c_ld, c_rd, c_col, start, j, k, l, maxsteps=5_000_000):
    """回载式循环：16B/层 int4 栈，重载父状态；无捕获位、无防重掩码。"""
    L, jkl, ld, rd, col0, last = _setup(N, c_ld, c_rd, c_col, start, j, k, l)
    init_pc = popcount(col0)
    leaf_pc = init_pc + (N - start - 2)
    jrow_base = start - init_pc
    valid = (~(ld | rd | col0 | jkl[start])) & M32 & last
    if valid == 0:
        return 0
    if start >= N - 2:
        return popcount(valid)
    stack = [(col0, ld, rd, valid)]
    sp = 1
    total = 0
    steps = 0
    while sp > 0:
        steps += 1
        if steps > maxsteps:
            return None
        cur, left, right, residual = stack[sp - 1]
        if residual == 0:
            sp -= 1
            continue
        p = (-residual & M32) & residual
        residual = (residual - p) & M32
        stack[sp - 1] = (cur, left, right, residual)
        cur2 = cur | p
        left2 = ((left | p) << 1) & M32
        right2 = ((right | p) >> 1) & M32
        pc = popcount(cur2)
        child = (~(cur2 | left2 | right2 | jkl[jrow_base + pc])) & M32 & last
        if pc == leaf_pc or child == 0:
            total += popcount(child)
        else:
            if sp == len(stack):
                stack.append((cur2, left2, right2, child))
            else:
                stack[sp] = (cur2, left2, right2, child)
            sp += 1
    return total


FRAME_RE = re.compile(
    r'FRAME ld=(\d+) rd=(\d+) col=(\d+) start=(\d+) j=(\d+) k=(\d+) l=(\d+)')


def load_frames(path, N):
    frames = []
    with open(path) as f:
        for line in f:
            m = FRAME_RE.match(line)
            if m:
                frames.append(tuple(int(x) for x in m.groups()))
    return [(N,) + fr for fr in frames]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    mode = 'both'
    if '--sim' in sys.argv:
        mode = sys.argv[sys.argv.index('--sim') + 1]
    if '--sum' in sys.argv:
        mode = 'sum'
    # N 从帧文件路径或环境推断：默认 20，可用 --N 指定
    N = 20
    if '--N' in sys.argv:
        N = int(sys.argv[sys.argv.index('--N') + 1])

    frames = load_frames(path, N)
    if not frames:
        print("no FRAME lines found in", path)
        return 1
    print(f"frames: {len(frames)}  N={N}")

    if mode == 'orig':
        for i, fr in enumerate(frames):
            print(f"CNT {i} {sim_orig(*fr)}")
        return 0
    if mode == 'reload':
        for i, fr in enumerate(frames):
            print(f"CNT {i} {sim_reload(*fr)}")
        return 0

    mism = 0
    sum_o = 0
    sum_r = 0
    for i, fr in enumerate(frames):
        so = sim_orig(*fr)
        sr = sim_reload(*fr)
        sum_o += so
        sum_r += sr
        if so != sr:
            mism += 1
            if mism <= 10:
                print(f"MISMATCH frame#{i}: ld={fr[1]} rd={fr[2]} col={fr[3]} "
                      f"start={fr[4]} j={fr[5]} k={fr[6]} l={fr[7]}  orig={so} reload={sr}")
    print(f"orig total = {sum_o}")
    print(f"reload total = {sum_r}")
    print(f"mismatches = {mism}")
    return 1 if mism else 0


if __name__ == '__main__':
    sys.exit(main())
