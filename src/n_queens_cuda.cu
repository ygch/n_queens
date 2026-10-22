// =============================================================================
// n_queens_cuda.cu — CUDA D4 星座法 N-Queens 求解器（回载式内核，本分支唯一内核）
//
// 整体架构：
//   1. CPU 端生成所有星座子问题（4个边界皇后 + preset_queens个预设皇后）
//   2. 按jkl分组排序，填充伪星座保证对齐（模 384 = LCM(32,64,96,128)）
//   3. 预计算每个block的jkl_queens对角线约束
//   4. GPU内核：每线程一个星座，int4 全状态重载式回溯（N=6~28 全部走回载内核）
//   5. CPU汇总：results[i] * symmetry(ijkl)，乘以D4对称倍数
//
// 内核状态表示：每层存完整状态 {cur,left,right,residual}（16B int4）；
//   回溯时整状态重载——不需要 ld_mem/rd_mem 捕获位，也不需要 direction 防重掩码；
//   叶子行(N-2) 用 __popc(child) 批量计数（倒数第二行优化）。
//
// 位掩码约定：
//   bit 0 = 最右列, bit N-1 = 最左列
//   行号: 0 (顶) ~ N-1 (底)
//
// 性能特征（RTX 4090，生产配置实测）：
//   N=20 N=20 block=128 ≈ 2152ms；N=21 block=32 ≈ 16.5s；N=22 block=64 ≈ 149s；N=23 block=32 ≈ 1352.4s
//   内核为吞吐量瓶颈（IPC限制），非延迟瓶颈；占用率受共享内存（精确栈深）限制
//
// 注：原始捕获式内核（4B/层栈 + 捕获位 + funnelshift）见 faf-cuda 分支
// =============================================================================

#include "constellation.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <vector>

// 块大小按 N 由 host 选择（见 cuda_solve 的按 N 表；默认 64，NQ_BLOCK 可覆盖）
#define MAX_BLOCK 128          // launch_bounds 上限

// GPU端星座数据结构（比CPU端Constellation更紧凑，无solutions字段）
// 16字节，便于全局内存合并访问
struct Constellation4 {
    int ld, rd, col, start_ijkl;
};

// =============================================================================
// GPU内核：D4星座 + Jeff Somers位并行回溯（回载循环 + 可抢占细粒度调度）
//
// 状态机（回载式）：
//   每层存完整状态 {cur,left,right,residual}（16B，int4 零 bank 冲突布局），
//   迭代时重载当前节点状态——不需要捕获位与防重掩码（对比 faf-cuda 捕获式）。
//
// 负载均衡（#56，2026-10-18）：
//   warp 内所有 lane 共用指令流，warp 要等最慢的 lane（N=20 实测迭代数 max/mean=1.52）。
//   这里把每个星座的 DFS 按 slice 次迭代切片：每轮每 lane 从块内任务队列领一个任务，
//   跑满 slice 次迭代后把 {栈, 部分解数, top} 检查点写回全局内存并重新入队；
//   未完成任务由任意 lane 继续（栈在 lane 私有 smem slab 中恢复）。
//   于是每一轮各 lane 工作量等长，warp 不再被单个大子树拖住。
//   任务池深度 = block_size*K（K 见 host；池会随任务完成缩小，K 太小后期 lane 会饿）。
//
// 计数与原内核逐子问题等价（Python 精确模拟对 1747 个真实子问题 0 误差验证）：
//   popc(cur)==leaf_pc ⇔ 刚放完第 N-3 行，子行(N-2)的合法位置数即解数
//   leaf_pc = popc(col) + (N - start - 2)；jkl 行号 = (start - popc(col)) + popc(cur)
// =============================================================================
template <int N>
__global__ void __launch_bounds__(MAX_BLOCK)
n_queens_kernel(
    const Constellation4 * __restrict__ constellations,  // 本block的 K*blockDim 个星座
    const unsigned int * __restrict__ jkl_queens_arr,     // 每block的N个jkl约束
    int S,                                                // 栈深：host 按真实最大深度精确计算
    long long * __restrict__ results,                     // 输出：每星座一个解数
    int K,                                                // 每block的星座数 = K*blockDim
    int slice,                                            // 每轮迭代预算（细粒度任务平衡）
    int4 * __restrict__ ckpt,                             // [星座槽 * S] 栈检查点
    int * __restrict__ ckpt_top,                          // 每任务已保存层数（0 = 未开始）
    long long * __restrict__ ckpt_sol,                    // 每任务已累计解数
    int * __restrict__ queue)                             // 每block：2*cap 环 + 3 计数器
{
    const int lid = threadIdx.x;
    const int nb = blockDim.x;
    const int cap = nb * K;
    const int myBase = blockIdx.x * cap;
    const unsigned int L = 1u << (N - 1);
    const unsigned int last = (1u << N) - 1;
    stk_guard_init:;

    // --- 协作加载 jkl_queens 到共享内存（整块共用一张表） ---
    __shared__ unsigned int jkl_queens[N];
    if (lid < N)
        jkl_queens[lid] = jkl_queens_arr[blockIdx.x * N + lid];

    // --- 共享内存栈：每线程 S 层 int4（lane l 的层 e 在 sbase + e*32） ---
    extern __shared__ int4 stk[];
    // 每 warp 的 smem 布局：[保护槽(1 层)] [栈 S 层]。
    // 保护槽在每任务开始时写为 {col, last, last, 0}：根节点弹栈后若还有半轮循环会读到它，
    // residual=0 不压栈、left|right=last ⇒ child=0；pc=popc(col)=init_pc ⇒ jkl 行号=start，恒合法。
    // 这样热循环可以删掉中途 top 检查（省 1 ALU + 1 分支）。
    const int sbase = (lid >> 5) * ((S + 1) * 32) + 32 + (lid & 31);

    // --- 每 block 一个任务队列（全局内存）：[0, 2*cap) 环形缓冲 + qhead/qtail/qpop ---
    int *q = queue + (size_t)blockIdx.x * (2 * cap + 4);
    int *qhead = q + 2 * cap;
    int *qtail = qhead + 1;
    int *qpop  = qhead + 2;

    if (lid == 0) { *qhead = 0; *qtail = 0; *qpop = 0; }
    __syncthreads();

    // 初始任务池：伪星座直接写 0，真实星座入队
    for (int i = lid; i < cap; i += nb) {
        int gid = myBase + i;
        if (constellations[gid].start_ijkl >> 20 == 69)
            results[gid] = 0;
        else
            q[atomicAdd(qtail, 1)] = gid;
    }

    // --- 主循环：每轮领一个任务、跑 slice 次迭代；未完成则检查点后回队 ---
    // lane 在任务边界被 warp 重新同步，但每轮工作量被预算截成等长 ⇒ warp 内不再等最慢子树
    for (;;) {
        __syncthreads();                       // (A) 上一轮的入队/账目可见
        const int cnt = *qtail - *qhead;       // 本轮可领任务数（稳定）
        if (cnt == 0)
            break;                             // 池空 ⇒ 全体一起退出
        const int take = atomicAdd(qpop, 1);
        const int task = (take < cnt) ? q[(*qhead + take) % (2 * cap)] : -1;

        bool finished = false;
        if (task >= 0) {
            const int gid = task;
            Constellation4 c = constellations[gid];
            const int start = c.start_ijkl >> 20;
            unsigned int ld = (unsigned int)c.ld;
            unsigned int rd = (unsigned int)c.rd;
            unsigned int col = (~(L - 2) ^ (unsigned int)c.col) & last;   // 裁到 N 位，高位存“死祖先计数”
            const int k_val = (c.start_ijkl >> 5) & 31;
            const int l_val = c.start_ijkl & 31;
            if (k_val >= start)
                ld &= ~(L >> (k_val - start));
            if (l_val != N - 1 && l_val >= start)
                rd &= ~(1u << (l_val - start));

            int top = 0;
            long long solutions = 0;
            const int t = ckpt_top[gid];
            if (t == 0) {
                unsigned int valid = last & ~(ld | rd | col | jkl_queens[start]);
                if (valid == 0) {              // 死根
                    results[gid] = 0;
                    finished = true;
                } else if (start >= N - 2) {   // 起始行即倒数第二行
                    results[gid] = __popc(valid);
                    finished = true;
                } else {
                    stk[sbase] = make_int4((int)col, (int)ld, (int)rd, (int)valid);
                    top = 1;
                }
            } else {                           // 恢复检查点
                top = t;
                solutions = ckpt_sol[gid];
                for (int e = 0; e < top; e++)
                    stk[sbase + e * 32] = ckpt[(size_t)gid * S + e];
            }

            if (!finished) {
                // 深度索引栈：槽位=深度 ⇒ 行号 = start+top，只依赖 top（两次 LDS 可并行发射）。
                // 压缩弹栈用“死祖先计数” skip 替代：压子时 skip'=耗尽?1+skip:0，弹栈 top-=1+skip。
                const int leaf_top = N - start - 2;
                char *sp = (char *)stk + (size_t)sbase * 16;   // 本 lane 槽 0 的字节地址
                stk[sbase - 32] = make_int4((int)col, (int)last, (int)last, 0);  // 保护槽
                int budget = slice;

                // --- 回溯循环（手工 unroll×2；预算到期即让出） ---
                // 手工 unroll×8：保护层在 #69 后是幂等的（弹栈 top=ret=0 保持不变），
                // 提前结束的 lane 在保护层上空转到组末，无需任何中途 top 检查。
                while (top > 0 && budget > 0) {
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    {
                    int4 *cslot = reinterpret_cast<int4 *>(sp + top * 512 - 512);
                    int4 e = *cslot;
                    unsigned int cur0 = (unsigned int)e.x & last;   // 去掉 ret 位
                    unsigned int right = (unsigned int)e.z;
                    unsigned int retplus;   // 本层弹栈后 top 应变成的值（“返回槽”）
                    if constexpr (N <= 27) {
                        retplus = (unsigned int)e.x >> N;                 // 5 位在 cur 高位
                    } else {
                        retplus = ((unsigned int)e.x >> N) | (((unsigned int)e.z >> 28) << 4);
                        right &= last;                                    // N=28：right 第 28 位存第 5 位
                    }
                    unsigned int left = (unsigned int)e.y;
                    unsigned int residual = (unsigned int)e.w;
                    // 行号只依赖 top → 与状态载入并行发射
                    const unsigned int jrow = jkl_queens[start + top];

                    unsigned int p = residual & (0u - residual);   // 最低可用位
                    residual -= p;
                    *(int *)((char *)cslot + 12) = (int)residual; // 存回残余候选
                    const bool exhausted = (residual == 0);

                    unsigned int cur = cur0 | p;
                    left = (left | p) << 1;
                    right = __umulhi(right | p, 0x80000000u);   // == (right|p)>>1，走 FMA 管

                    unsigned int occ, child;
                    asm("lop3.b32 %0, %1, %2, %3, 0xfe;" : "=r"(occ) : "r"(cur), "r"(left), "r"(right));
                    asm("lop3.b32 %0, %1, %2, %3, 0x02;" : "=r"(child) : "r"(occ), "r"(jrow), "r"(last));

                    const bool push = (top != leaf_top) && (child != 0);
                    const int ntop = push ? top + 1 : (exhausted ? (int)retplus : top);
                    if (push) {
                        // 父层活着 → 子层弹回父槽（ret'=top）；父层已死 → 继承父层的返回槽
                        const unsigned int nret = exhausted ? retplus : (unsigned int)top;
                        unsigned int ncur = cur, nright = right;
                        if constexpr (N <= 27) {
                            ncur = cur | (nret << N);
                        } else {
                            ncur = cur | ((nret & 15u) << N);
                            nright = right | ((nret >> 4) << N);
                        }
                        // 压栈槽 = 当前槽 + 1（cslot+512），复用当前槽地址的立即偏移，无需另算地址
                        *reinterpret_cast<int4 *>((char *)cslot + 512) = make_int4((int)ncur, (int)left, (int)nright, (int)child);
                    } else {
                        // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
                        solutions += __popc(child);
                    }
                    top = ntop;
                    }
                    budget -= 8;
                }

                finished = (top <= 0);
                if (finished) {
                    results[gid] = solutions;
                } else {                           // 保存检查点
                    for (int e = 0; e < top; e++)
                        ckpt[(size_t)gid * S + e] = stk[sbase + e * 32];
                    ckpt_sol[gid] = solutions;
                    ckpt_top[gid] = top;
                }
            }
        }

        __syncthreads();                       // (B) 所有领取/时间片结束；qhead 更新互斥
        if (task >= 0 && !finished) {
            int pos = atomicAdd(qtail, 1);
            q[pos % (2 * cap)] = task;         // 未完成任务回队
        }
        if (lid == 0) {
            const int used = (*qpop < cnt) ? *qpop : cnt;
            *qhead += used;
            *qpop = 0;
        }
    }
}

// =============================================================================
// Host 端辅助函数
// =============================================================================

// 按jkl排序星座：相同jkl的星座分到同一block，共享jkl_queens共享内存
static void sortByJkl(std::vector<Constellation> &consts) {
    std::sort(consts.begin(), consts.end(), [](const Constellation &a, const Constellation &b) {
        return getJkl(a.start_ijkl) < getJkl(b.start_ijkl);
    });
}

// 填充伪星座：在每个jkl组之间插入假数据，使每组星座数为blockSize的倍数
// 确保每个block内所有线程处理相同jkl的星座
// 伪星座标记：start_ijkl = (69 << 20)，内核检测到start==69直接返回0
static void padWithPseudo(std::vector<Constellation> &consts, int n, int blockSize) {
    sortByJkl(consts);
    std::vector<Constellation> padded;
    int allOnes = (1 << n) - 1;
    int currentJkl = getJkl(consts[0].start_ijkl);
    for (auto &c : consts) {
        if (getJkl(c.start_ijkl) != currentJkl) {
            // jkl组切换：填充伪星座直到对齐blockSize
            while (padded.size() % blockSize != 0)
                padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
            currentJkl = getJkl(c.start_ijkl);
        }
        padded.push_back(c);
    }
    // 末尾对齐
    while (padded.size() % blockSize != 0)
        padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
    consts = std::move(padded);
}

// 宏：为每个N值生成一个switch case分支
// 1. 请求最大共享内存carveout（RTX 4090为102KB）
// 2. 启动模板化的内核：<<<num_blocks, block_size>>>
#define LAUNCH_ONE(n_val, KERNEL, SMEM) \
    case n_val: { \
        cudaFuncSetAttribute(KERNEL<n_val>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
        if (getenv("NQ_OCC")) { \
            int occ_ = 0; \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_, KERNEL<n_val>, block_size, (size_t)(SMEM)); \
            fprintf(stderr, "[occ] n=%d block=%d smem=%zu -> blocks/SM=%d warps/SM=%d\n", \
                    n_val, (int)block_size, (size_t)(SMEM), occ_, occ_ * (int)block_size / 32); \
        } \
        KERNEL<n_val><<<num_blocks, block_size, SMEM>>>(d_const, d_jkl, maxd, d_results, K, \
                                                       slice, d_ckpt, d_ctop, d_csol, d_queue); \
        break; \
    }

// 本分支唯一内核：回载式（动态共享内存 = 精确栈深）；N=6~28 全部走回载
// 同会话 A/B 实测：N=19~21 快 12~13%；N=22 快 8.1%（149.1s vs 162.2s）；N=23 快 8.0%（1352.4s vs 1469.7s）
// （捕获式内核见 faf-cuda 分支）
#define LAUNCH_N(n_val) LAUNCH_ONE(n_val, n_queens_kernel, smem_reload)

// =============================================================================
// 多卡求解（mod 交错分配）
// 分配粒度 = block：同一 block 的星座共享一份 jkl_queens 表，
// block（编号 blk）归卡 blk % gpu_num；各卡把分到的 block 重排为连续段后，
// 走与单卡相同的内核（段内 blockIdx 与重排后的 jkl 行一一对应）。
// =============================================================================
static long long solve_multi_gpu(int n, int gpu_num, int block_size, int K, int num_blocks, int maxd,
                                 const std::vector<Constellation4> &const_data,
                                 const std::vector<uint32_t> &jkl_queens_arr) {
    struct timeval t0, t1;
    long long sum = 0;

    std::vector<int> block_cnt(gpu_num, 0);
    for (int blk = 0; blk < num_blocks; blk++)
        block_cnt[blk % gpu_num]++;

    std::vector<std::vector<Constellation4>> g_const(gpu_num);
    std::vector<std::vector<uint32_t>> g_jkl(gpu_num);
    std::vector<std::vector<int>> g_w(gpu_num);   // symmetry 权重（伪星座记 0）
    for (int g = 0; g < gpu_num; g++) {
        g_const[g].reserve((size_t)block_cnt[g] * block_size * K);
        g_jkl[g].reserve((size_t)block_cnt[g] * n);
        g_w[g].reserve((size_t)block_cnt[g] * block_size * K);
    }
    for (int blk = 0; blk < num_blocks; blk++) {
        int g = blk % gpu_num;
        const Constellation4 *src = const_data.data() + (size_t)blk * block_size * K;
        g_const[g].insert(g_const[g].end(), src, src + (size_t)block_size * K);
        for (int j = 0; j < block_size * K; j++) {
            int s = src[j].start_ijkl >> 20;
            g_w[g].push_back(s == 69 ? 0 : symmetry(n, src[j].start_ijkl & 0xFFFFF));
        }
        const uint32_t *row = jkl_queens_arr.data() + (size_t)blk * n;
        g_jkl[g].insert(g_jkl[g].end(), row, row + n);
    }

    std::vector<std::vector<long long>> g_res(gpu_num);
    gettimeofday(&t0, NULL);
#pragma omp parallel num_threads(std::min(gpu_num, num_blocks))
    {
        int g = omp_get_thread_num();
        cudaSetDevice(g);

        int num_blocks = block_cnt[g];                        // 宏内变量名对齐（本卡）
        long long my_cnt = (long long)g_const[g].size();
        print_with_time("gpu [%d] start job, with %d blocks (%lld subproblems).\n",
                        g, num_blocks, my_cnt);

        Constellation4 *d_const;
        unsigned int *d_jkl;
        long long *d_results;
        cudaMalloc(&d_const, my_cnt * sizeof(Constellation4));
        cudaMalloc(&d_jkl, g_jkl[g].size() * sizeof(uint32_t));
        cudaMalloc(&d_results, my_cnt * sizeof(long long));
        cudaMemcpy(d_const, g_const[g].data(), my_cnt * sizeof(Constellation4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jkl, g_jkl[g].data(), g_jkl[g].size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_results, 0, my_cnt * sizeof(long long));

        int slice = 16384;
        { const char *e = getenv("NQ_SLICE"); if (e) slice = atoi(e); if (slice < 2) slice = 2; }
        int4 *d_ckpt;
        int *d_ctop;
        long long *d_csol;
        int *d_queue;
        cudaMalloc(&d_ckpt, (size_t)my_cnt * maxd * sizeof(int4));
        cudaMalloc(&d_ctop, (size_t)my_cnt * sizeof(int));
        cudaMalloc(&d_csol, (size_t)my_cnt * sizeof(long long));
        cudaMalloc(&d_queue, (size_t)num_blocks * (2 * block_size * K + 4) * sizeof(int));
        cudaMemset(d_ctop, 0, (size_t)my_cnt * sizeof(int));

        size_t smem_reload = (size_t)(block_size / 32) * (maxd + 1) * 32 * sizeof(int4);
    { const char *e = getenv("NQ_SMEM_PAD"); if (e) smem_reload += atoi(e); }   // 实验：模拟更大 N 的每块 smem   // 宏内变量名对齐（+1 保护槽）
        switch (n) {
            LAUNCH_N(6)  LAUNCH_N(7)  LAUNCH_N(8)  LAUNCH_N(9)  LAUNCH_N(10)
            LAUNCH_N(11) LAUNCH_N(12) LAUNCH_N(13) LAUNCH_N(14) LAUNCH_N(15)
            LAUNCH_N(16) LAUNCH_N(17) LAUNCH_N(18) LAUNCH_N(19) LAUNCH_N(20)
            LAUNCH_N(21) LAUNCH_N(22) LAUNCH_N(23) LAUNCH_N(24) LAUNCH_N(25)
            LAUNCH_N(26) LAUNCH_N(27) LAUNCH_N(28)
            default:
                print_with_time("Unsupported N: %d\n", n);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            print_with_time("CUDA kernel error: %s\n", cudaGetErrorString(err));
        }

        g_res[g].resize(my_cnt);
        cudaMemcpy(g_res[g].data(), d_results, my_cnt * sizeof(long long), cudaMemcpyDeviceToHost);

        cudaFree(d_const);
        cudaFree(d_jkl);
        cudaFree(d_results);
        cudaFree(d_ckpt);
        cudaFree(d_ctop);
        cudaFree(d_csol);
        cudaFree(d_queue);
        std::vector<Constellation4>().swap(g_const[g]);   // 及时释放主机内存
        std::vector<uint32_t>().swap(g_jkl[g]);
        print_with_time("gpu [%d] finish job.\n", g);
    }
    gettimeofday(&t1, NULL);
    print_with_time("CUDA kernel time: %.2fms\n", time_diff_ms(t0, t1));

    // 汇总（与单卡同口径：results × symmetry；伪星座权重 0）
    for (int g = 0; g < gpu_num; g++)
        for (size_t j = 0; j < g_res[g].size(); j++)
            sum += g_res[g][j] * g_w[g][j];

    return sum;
}

// =============================================================================
// cuda_solve — 主入口函数
//
// 流程：
//   1. CPU生成星座子问题
//   2. 排序填充对齐
//   3. 预计算jkl_queens（边界皇后逐行对角线约束）
//   4. 传输数据到GPU
//   5. switch(N)启动对应模板内核
//   6. 拷回结果，汇总 sum += results[i] * symmetry(ijkl)
// =============================================================================
long long cuda_solve(int n, int preset_queens) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    // Step 1: CPU端生成所有星座子问题
    auto constellations = generate_constellations(n, preset_queens);
    gettimeofday(&t1, NULL);
    print_with_time("Constellation generation: %.2fms, %zu subproblems\n",
                    time_diff_ms(t0, t1), constellations.size());

    if (constellations.empty()) {
        print_with_time("No subproblems generated!\n");
        return 0;
    }

    // 按 N 的块大小表（队列版实测最优；默认 64）
    int block_size = 64;
    if (n == 20)
        block_size = 128;        // N=20 实测 1428ms（b=32 1576ms、b=64 1614ms）
    else if (n == 21 || n >= 23)
        block_size = 32;         // N=21 实测 12.46s vs 64 的 12.81s
                                // 例外 N=22（走默认 64）：实测 108.3s vs b=32 的 116.2s
    // N=21 r7（16 层）：b=32 只驻留 10 块 = 10 warps；b=128 的 32,768B 可驻留 3 块 × 4 = 12 warps
    // （cudaOccupancy 实测）。实测 8229ms vs 8593ms（−4.2%）。r6 是 17 层，b=128 只剩 8 warps，仍用 b=32。
    if (n == 21 && preset_queens >= 7)
        block_size = 128;
    // N=23：r7（18 层）b=32 只有 9 块/SM；b=64（2 warps/块）→ 10 块 = 10 warps。
    // 实测（8×5090，K=16/16384）：窗口 96.6s → 89.8s（−7.0%），计数正确。r6 仍是 b=32 最优。
    if (n == 23 && preset_queens >= 7)
        block_size = 64;

    // N=28（栈 23/24 层）：b=32 只能驻留 7 块/SM。按可用 smem≈99KB：
    //   r7（23 层）：b=64 → 4 块 × 2 = 8 warps（+14%）
    //   r6（24 层）：b=64 → 仅 3 块 = 6 warps（更差！），b=128 → 2 块 × 4 = 8 warps
    if (n == 28)
        block_size = (preset_queens >= 7) ? 64 : 128;

    // N=25/26 r7（栈 20/21 层）：单 warp 块只能驻留 8 块（8 warps）。把每块 1KB 固定开销
    // 摊到 3 个 warp 上：3 块 × 3 warp = 9 warps/SM（cudaOccupancy API 实测；N=26 的
    // 32,256B 动态 smem 用 N=25 + pad 模拟验证，同为 3 块）。3 warp 同块有轮次耦合：
    // N=22 代理实测 104.6s vs 8 个独立 warp 108.4s = 净 +3.5%（9 独立 warp 为 99.0s），
    // 任务越大每轮“有 lane 提前结束”的概率越低，N=25/26 的耦合应更小。
    if ((n == 25 || n == 26) && preset_queens >= 7)
        block_size = 96;
    {   // 实验开关：NQ_BLOCK 环境变量可覆盖按 N 表（便于扫描，不影响正常使用）
        const char *bs_env = getenv("NQ_BLOCK");
        if (bs_env)
            block_size = atoi(bs_env);
    }
    // 每 block 的任务数 = block_size*K。抢占式调度下 K 决定任务池深度：
    // 池随任务完成而缩小，K 越大池保持"满 lane"的时间越久（#56 实测 K=4~6 最优）。
    // K（每 block 的任务数 = block_size*K）。r7 的任务比 r6 小 8–10×，
    // 池深取 16 可缩短队列抽干时的尾部（N=22 r7 实测 −2.5%）；r6 仍取 6。
    int K = (preset_queens >= 7) ? 16 : 6;
    {
        const char *k_env = getenv("NQ_K");
        if (k_env)
            K = atoi(k_env);
        if (K < 1) K = 1;
    }

    // Step 2: 按jkl排序、填充伪星座保证对齐
    // 对齐模数 = block_size*K：每个 block 处理 K*blockDim 个同 jkl 的星座
    padWithPseudo(constellations, n, block_size * K);
    long long cnt = constellations.size();

    int num_blocks = (int)(cnt / ((long long)block_size * K));

    print_with_time("Total constellations (padded): %lld, blocks: %d\n", cnt, num_blocks);

    // Step 3: 预计算 jkl_queens_arr
    // 对每个block，根据其首个星座的ijkl计算边界皇后j/k/l在每一行的对角线约束
    // jkl_queens[row] 是一个位掩码，标记该行被j/k/l对角线占用的列
    unsigned int L = 1u << (n - 1);
    std::vector<uint32_t> jkl_queens_arr(num_blocks * n, 0);
    for (int blk = 0; blk < num_blocks; blk++) {
        // 取block中第一个星座的ijkl（同组所有线程共享）
        int ijkl = constellations[(long long)blk * block_size * K].start_ijkl & 0xFFFFF;
        int j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

        // 第一轮：计算j和k/l从棋盘底部向顶部延伸的对角线
        // rdiag: j的右对角线 + k的右对角线（从底向上）
        unsigned int rdiag = (L >> j) | (L >> (n - 1 - k));
        // ldiag: j的左对角线 + l的左对角线（从底向上）
        unsigned int ldiag = (L >> j) | (L >> l);
        for (int row = 0; row < n; row++)
            jkl_queens_arr[blk * n + (n - 1 - row)] = (ldiag >> row) | (rdiag << row) | L | 1;
            // | L | 1: 设置两个边界列（bit 0 和 bit N-1），因为l和k皇后占据这两列

        // 第二轮：叠加k和l从顶部向底部延伸的对角线
        ldiag = L >> k;   // k的左对角线（从顶向下）
        rdiag = 1u << l;  // l的右对角线（从顶向下）
        for (int row = 0; row < n; row++)
            jkl_queens_arr[blk * n + row] |= (ldiag << row) | (rdiag >> row);

        // 边界皇后所在行的特殊处理：
        // k行：第一列已有皇后，清除bit N-1（左边界）避免冲突
        jkl_queens_arr[blk * n + k] = ~L;
        // l行：最后一列已有皇后，清除bit 0（右边界）避免冲突
        jkl_queens_arr[blk * n + l] = ~1u;
    }

    // Step 3.6: 回载内核的精确栈深。
    // 内核在 row==N-3 的叶子层只计数、不压栈，因此最大 top = (N-2)-start（不是 +1）。
    // 槽数按此精确分配；每省 1 层 = 每 warp 少 512B smem（可能多一个驻留块）。
    int maxd = 1;   // 槽数
    for (long long i = 0; i < cnt; i++) {
        int s = constellations[i].start_ijkl >> 20;
        if (s == 69 || s >= n - 2)
            continue;
        int d = (n - 2) - s;          // 叶子层不压栈，真实最大 top
        if (d > maxd)
            maxd = d;
    }
    // 每 warp 分配 maxd 层栈 + 1 个保护槽（放在栈下方，用于删除热循环的中途退出检查）
    size_t smem_reload = (size_t)(block_size / 32) * (maxd + 1) * 32 * sizeof(int4);

    // Step 4: 转换为GPU端紧凑格式
    std::vector<Constellation4> const_data(cnt);
    for (long long i = 0; i < cnt; i++) {
        const_data[i] = {constellations[i].ld, constellations[i].rd,
                         constellations[i].col, constellations[i].start_ijkl};
    }

    // 多卡：block 粒度 mod 交错分配（blk 归卡 blk % gpu_num）
    int gpu_num = 0;
    cudaGetDeviceCount(&gpu_num);
    if (gpu_num > 1)
        return solve_multi_gpu(n, gpu_num, block_size, K, num_blocks, maxd, const_data, jkl_queens_arr);

    // Step 5: 分配GPU内存并传输数据
    Constellation4 *d_const;
    unsigned int *d_jkl;
    long long *d_results;
    cudaMalloc(&d_const, cnt * sizeof(Constellation4));
    cudaMalloc(&d_jkl, num_blocks * n * sizeof(unsigned int));
    cudaMalloc(&d_results, cnt * sizeof(long long));

    cudaMemcpy(d_const, const_data.data(), cnt * sizeof(Constellation4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jkl, jkl_queens_arr.data(), num_blocks * n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, cnt * sizeof(long long));

    // 细粒度任务平衡（#56）：每星座栈检查点 + 每 block 任务队列
    int slice = 16384;
    { const char *e = getenv("NQ_SLICE"); if (e) slice = atoi(e); if (slice < 2) slice = 2; }
    int4 *d_ckpt;
    int *d_ctop;
    long long *d_csol;
    int *d_queue;
    cudaMalloc(&d_ckpt, (size_t)cnt * maxd * sizeof(int4));
    cudaMalloc(&d_ctop, (size_t)cnt * sizeof(int));
    cudaMalloc(&d_csol, (size_t)cnt * sizeof(long long));
    cudaMalloc(&d_queue, (size_t)num_blocks * (2 * block_size * K + 4) * sizeof(int));
    cudaMemset(d_ctop, 0, (size_t)cnt * sizeof(int));
    print_with_time("checkpoint buffers: %.0f MB (slice=%d)\n",
                    ((double)cnt * maxd * 16 + (double)cnt * 12) / 1048576.0, slice);
    if (!d_ckpt || !d_ctop || !d_csol || !d_queue) {
        print_with_time("checkpoint allocation failed (out of GPU memory)\n");
        cudaFree(d_ckpt); cudaFree(d_ctop); cudaFree(d_csol); cudaFree(d_queue);
        cudaFree(d_const); cudaFree(d_jkl); cudaFree(d_results);
        return -1;
    }

    // Step 6: 启动GPU内核
    gettimeofday(&t0, NULL);
    switch (n) {
        LAUNCH_N(6)  LAUNCH_N(7)  LAUNCH_N(8)  LAUNCH_N(9)  LAUNCH_N(10)
        LAUNCH_N(11) LAUNCH_N(12) LAUNCH_N(13) LAUNCH_N(14) LAUNCH_N(15)
        LAUNCH_N(16) LAUNCH_N(17) LAUNCH_N(18) LAUNCH_N(19) LAUNCH_N(20)
        LAUNCH_N(21) LAUNCH_N(22) LAUNCH_N(23)
        LAUNCH_N(24) LAUNCH_N(25)
        LAUNCH_N(26) LAUNCH_N(27) LAUNCH_N(28)
        default:
            print_with_time("Unsupported N: %d\n", n);
            cudaFree(d_const); cudaFree(d_jkl); cudaFree(d_results);
            return -1;
    }
    cudaError_t err = cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    if (err != cudaSuccess) {
        print_with_time("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    print_with_time("CUDA kernel time: %.2fms\n", time_diff_ms(t0, t1));

    // Step 7: 拷回结果
    std::vector<long long> results(cnt);
    cudaMemcpy(results.data(), d_results, cnt * sizeof(long long), cudaMemcpyDeviceToHost);

    {   // 实验开关：导出每线程解数，用于分析 warp 内负载分布（不影响正常运行）
        const char *dump = getenv("NQ_DUMP");
        if (dump) {
            FILE *f = fopen(dump, "wb");
            if (f) { fwrite(results.data(), sizeof(long long), cnt, f); fclose(f); }
            char gp[512];
            snprintf(gp, sizeof(gp), "%s.ijkl", dump);
            FILE *g = fopen(gp, "wb");
            if (g) {
                for (long long i = 0; i < cnt; i++) {
                    int v = constellations[i].start_ijkl;
                    fwrite(&v, sizeof(int), 1, g);
                }
                fclose(g);
            }
        }
    }


    cudaFree(d_const);
    cudaFree(d_jkl);
    cudaFree(d_results);
    cudaFree(d_ckpt);
    cudaFree(d_ctop);
    cudaFree(d_csol);
    cudaFree(d_queue);

    // Step 8: 汇总——每个星座的解数乘以D4对称倍数
    // symmetry()返回8（一般情况）/4（180度对称）/2（90度对称）
    long long sum = 0;
    for (long long i = 0; i < cnt; i++) {
        if (constellations[i].start_ijkl >> 20 == 69)
            continue;  // 跳过伪星座
        int ijkl = constellations[i].start_ijkl & 0xFFFFF;
        sum += results[i] * symmetry(n, ijkl);
    }

    return sum;
}
