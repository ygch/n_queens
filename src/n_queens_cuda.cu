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
// GPU内核：D4星座 + Jeff Somers位并行回溯（回载式循环）
//
// 与旧循环的区别：
//   - 旧：每层只存 queen(4B)，回溯用"反向移位 + ld_mem/rd_mem 捕获位"重建
//         ld/rd，并依赖 direction 防重掩码；每次迭代约 15-20 条 SASS。
//   - 新：每层存完整状态 {cur,left,right,residual}（16B，int4 零 bank 冲突布局），
//         迭代时重载父状态——不再需要捕获位与防重掩码。
//
// 计数与原内核逐子问题等价（Python 精确模拟对 1747 个真实子问题 0 误差验证）：
//   popc(cur)==leaf_pc ⇔ 刚放完第 N-3 行，子行(N-2)的合法位置数即解数
//   leaf_pc = popc(col) + (N - start - 2)；jkl 行号 = (start - popc(col)) + popc(cur)
// =============================================================================
template <int N>
__global__ void __launch_bounds__(MAX_BLOCK, 12)
n_queens_kernel(
    const Constellation4 * __restrict__ constellations,  // 每线程一个星座
    const unsigned int * __restrict__ jkl_queens_arr,     // 每block的N个jkl约束
    int S,                                                // 栈深：host 按真实最大深度精确计算
    long long * __restrict__ results)                     // 输出：每线程一个解数
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // --- 1. 加载星座数据到寄存器 ---
    Constellation4 c = constellations[gid];
    const unsigned int L = 1u << (N - 1);
    unsigned int ld = (unsigned int)c.ld;    // 到 start 行对齐的对角线状态
    unsigned int rd = (unsigned int)c.rd;
    unsigned int col = ~(L - 2) ^ (unsigned int)c.col;

    // --- 2. 协作加载 jkl_queens 到共享内存 ---
    __shared__ unsigned int jkl_queens[N];
    if (lid < N)
        jkl_queens[lid] = jkl_queens_arr[blockIdx.x * N + lid];
    __syncthreads();

    // --- 3. 处理伪星座 ---
    int start = c.start_ijkl >> 20;
    if (start == 69) {
        results[gid] = 0;
        return;
    }

    // --- 4. 修正边界皇后k和l的对角线（与旧内核一致） ---
    int k_val = (c.start_ijkl >> 5) & 31;
    int l_val = c.start_ijkl & 31;
    if (k_val >= start)
        ld &= ~(L >> (k_val - start));
    if (l_val != N - 1 && l_val >= start)
        rd &= ~(1u << (l_val - start));

    // --- 5. 初始候选与短路 ---
    const unsigned int last = (1u << N) - 1;
    unsigned int valid = last & ~(ld | rd | col | jkl_queens[start]);
    if (valid == 0) {          // 死根：原内核同样立即弹栈退出
        results[gid] = 0;
        return;
    }
    if (start >= N - 2) {      // 起始行即倒数第二行：合法位置数即解数
        results[gid] = __popc(valid);
        return;
    }

    // 回载式循环的换算常数（每子问题仅算一次）
    const int init_pc = __popc(col);
    const int leaf_pc = init_pc + (N - start - 2);
    const int jrow_base = start - init_pc;

    // --- 6. 共享内存栈：每线程 S 层 int4，零 bank 冲突布局 ---
    // 同 warp 内 lane l 的层 e 位于槽 e*32+l；连续 lane 的 16B 向量铺满 32 bank
    // S 由 host 按真实最大栈深精确给出（动态共享内存），避免按 N-3 过度分配压低占用率
    extern __shared__ int4 stk[];
    const int sbase = (lid >> 5) * (S * 32) + (lid & 31);

    stk[sbase] = make_int4((int)col, (int)ld, (int)rd, (int)valid);
    int top = 1;               // 层 e 位于槽 sbase + (e-1)*32
    long long solutions = 0;

    // --- 7. 主回溯循环（手工 unroll×2：两份作用域化的循环体 + 中途退出） ---
    while (top > 0) {
        {
        int4 e = stk[sbase + (top - 1) * 32];
        unsigned int cur = (unsigned int)e.x;
        unsigned int left = (unsigned int)e.y;
        unsigned int right = (unsigned int)e.z;
        unsigned int residual = (unsigned int)e.w;

        unsigned int p = residual & (0u - residual);   // 最低可用位
        residual -= p;
        stk[sbase + (top - 1) * 32].w = (int)residual; // 存回残余候选
        if (residual == 0)
            top--;                                     // 本层耗尽 → 弹栈

        cur |= p;
        left = (left | p) << 1;
        right = (right | p) >> 1;

        int pc = __popc(cur);
        unsigned int child =
            last & ~(cur | left | right | jkl_queens[jrow_base + pc]);

        if (pc == leaf_pc || child == 0) {
            // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
            solutions += __popc(child);
        } else {
            stk[sbase + top * 32] = make_int4((int)cur, (int)left, (int)right, (int)child);
            top++;
        }
        }
        if (top <= 0)
            break;
        {
        int4 e = stk[sbase + (top - 1) * 32];
        unsigned int cur = (unsigned int)e.x;
        unsigned int left = (unsigned int)e.y;
        unsigned int right = (unsigned int)e.z;
        unsigned int residual = (unsigned int)e.w;

        unsigned int p = residual & (0u - residual);   // 最低可用位
        residual -= p;
        stk[sbase + (top - 1) * 32].w = (int)residual; // 存回残余候选
        if (residual == 0)
            top--;                                     // 本层耗尽 → 弹栈

        cur |= p;
        left = (left | p) << 1;
        right = (right | p) >> 1;

        int pc = __popc(cur);
        unsigned int child =
            last & ~(cur | left | right | jkl_queens[jrow_base + pc]);

        if (pc == leaf_pc || child == 0) {
            // 叶子行(N-2)：child 的 popc 即解数；死胡同 child==0 加 0
            solutions += __popc(child);
        } else {
            stk[sbase + top * 32] = make_int4((int)cur, (int)left, (int)right, (int)child);
            top++;
        }
        }
    }

    results[gid] = solutions;
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
        KERNEL<n_val><<<num_blocks, block_size, SMEM>>>(d_const, d_jkl, maxd, d_results); \
        break; \
    }

// 本分支唯一内核：回载式（动态共享内存 = 精确栈深）；N=6~28 全部走回载
// 同会话 A/B 实测：N=19~21 快 12~13%；N=22 快 8.1%（149.1s vs 162.2s）；N=23 快 8.0%（1352.4s vs 1469.7s）
// （捕获式内核见 faf-cuda 分支）
#define LAUNCH_N(n_val) LAUNCH_ONE(n_val, n_queens_kernel, smem_reload)

// =============================================================================
// 多卡求解（mod 交错分配）
// 分配粒度 = block：同一 block 的星座共享一份 jkl_queens 表，
// block blk 归卡 (blk % gpu_num)；各卡把分到的 block 重排为连续段后，
// 走与单卡相同的内核（段内 blockIdx 与重排后的 jkl 行一一对应）。
// =============================================================================
static long long solve_multi_gpu(int n, int gpu_num, int block_size, int num_blocks, int maxd,
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
        g_const[g].reserve((size_t)block_cnt[g] * block_size);
        g_jkl[g].reserve((size_t)block_cnt[g] * n);
        g_w[g].reserve((size_t)block_cnt[g] * block_size);
    }
    for (int blk = 0; blk < num_blocks; blk++) {
        int g = blk % gpu_num;
        const Constellation4 *src = const_data.data() + (size_t)blk * block_size;
        g_const[g].insert(g_const[g].end(), src, src + block_size);
        for (int j = 0; j < block_size; j++) {
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

        size_t smem_reload = (size_t)(block_size / 32) * maxd * 32 * sizeof(int4);   // 宏内变量名对齐
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

    // Step 2: 按jkl排序、填充伪星座保证对齐
    // 填充模数 384 = LCM(32,64,96,128)：任何按 N 选择的块大小都能整除组边界
    padWithPseudo(constellations, n, 384);
    long long cnt = constellations.size();

    // 按 N 的块大小表（实测最优；默认 64）
    int block_size = 64;
    if (n == 20)
        block_size = 128;        // N=20 实测 2195ms vs 64 的 2354ms
    else if (n == 21 || n >= 23)
        block_size = 32;         // N=21 实测 16.72s vs 64 的 17.78s；N=23 实测 1352.4s
                                // N=24~28：κ 模型预测 32 最优（未实测，跑 N=24 前先验证）
    {   // 实验开关：NQ_BLOCK 环境变量可覆盖按 N 表（便于扫描，不影响正常使用）
        const char *bs_env = getenv("NQ_BLOCK");
        if (bs_env)
            block_size = atoi(bs_env);
    }
    int num_blocks = (int)(cnt / block_size);

    print_with_time("Total constellations (padded): %lld, blocks: %d\n", cnt, num_blocks);

    // Step 3: 预计算 jkl_queens_arr
    // 对每个block，根据其首个星座的ijkl计算边界皇后j/k/l在每一行的对角线约束
    // jkl_queens[row] 是一个位掩码，标记该行被j/k/l对角线占用的列
    unsigned int L = 1u << (n - 1);
    std::vector<uint32_t> jkl_queens_arr(num_blocks * n, 0);
    for (int blk = 0; blk < num_blocks; blk++) {
        // 取block中第一个星座的ijkl（同组所有线程共享）
        int ijkl = constellations[blk * block_size].start_ijkl & 0xFFFFF;
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

    // Step 3.6: 回载内核的精确栈深（槽数 = (N-2)-start + 1；跳过伪星座与短路子问题）
    int maxd = 1;   // 槽数
    for (long long i = 0; i < cnt; i++) {
        int s = constellations[i].start_ijkl >> 20;
        if (s == 69 || s >= n - 2)
            continue;
        int d = (n - 2) - s + 1;
        if (d > maxd)
            maxd = d;
    }
    size_t smem_reload = (size_t)(block_size / 32) * maxd * 32 * sizeof(int4);   // 回载内核：int4 栈（精确深度）

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
        return solve_multi_gpu(n, gpu_num, block_size, num_blocks, maxd, const_data, jkl_queens_arr);

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


    cudaFree(d_const);
    cudaFree(d_jkl);
    cudaFree(d_results);

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
