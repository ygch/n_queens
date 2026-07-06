// =============================================================================
// n_queens_cuda.cu — CUDA D4 星座法 N-Queens 求解器
//
// 整体架构：
//   1. CPU 端生成所有星座子问题（4个边界皇后 + preset_queens个预设皇后）
//   2. 按jkl分组排序，填充伪星座保证对齐
//   3. 预计算每个workgroup的jkl_queens对角线约束
//   4. GPU内核：每个线程一个星座，Jeff Somers位并行回溯
//   5. CPU汇总：results[i] * symmetry(ijkl)，乘以D4对称倍数
//
// 核心算法：Jeff Somers位并行回溯
//   用3个32位掩码表示棋盘状态：
//     col — 列占用 (bit c = 1 表示列c已有皇后)
//     ld  — 左对角线 (\方向)，下一行整体左移1位 (<<1)
//     rd  — 可对角线 (/方向)，下一行整体右移1位 (>>1)
//   当32位移位导致位溢出时，用ld_mem/rd_mem捕获丢失的位
//
// 位掩码约定：
//   bit 0 = 最右列, bit N-1 = 最左列
//   行号: 0 (顶) ~ N-1 (底)
//
// 性能特征（RTX 4090, N=20）：
//   内核时间 ~2329ms, 20寄存器, 5200字节共享内存, 9块/SM
//   内核为吞吐量瓶颈（IPC限制），非延迟瓶颈
// =============================================================================

#include "constellation.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

// 每个线程块的线程数（2个warp）
// 较小的块大小使每个SM能容纳更多块，增加warp调度灵活性
#define WG_SIZE 64

// GPU端星座数据结构（比CPU端Constellation更紧凑，无solutions字段）
// 16字节，便于全局内存合并访问
struct Constellation4 {
    int ld, rd, col, start_ijkl;
};

// =============================================================================
// GPU内核：Jeff Somers位并行回溯
//
// 模板参数N：棋盘大小。模板化使编译器将N作为编译时常量优化
// （如 1<<(N-1) 变为立即数），并为每个N生成最优代码。
//
// __launch_bounds__(64, 24)：告诉编译器每块64线程，目标每SM 24块。
// 实际受共享内存限制为9块/SM（18 warp）。
//
// 每个线程处理一个独立的星座子问题，线程间无数据依赖。
// =============================================================================
template <int N>
__global__ void __launch_bounds__(WG_SIZE, 24)
n_queens_kernel(
    const Constellation4 * __restrict__ constellations,  // 每线程一个星座
    const unsigned int * __restrict__ jkl_queens_arr,     // 每workgroup的N个jkl约束
    long long * __restrict__ results)                     // 输出：每线程一个解数
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // --- 1. 加载星座数据到寄存器 ---
    Constellation4 c = constellations[gid];

    // L = 最高位掩码（bit N-1），用于标记左右边界列
    const unsigned int L = 1u << (N - 1);

    // ld/rd/col：棋盘状态掩码，来自CPU端预设皇后放置后的状态
    unsigned int ld = (unsigned int)c.ld;
    unsigned int rd = (unsigned int)c.rd;

    // col初始化：~(L-2) 设置两个边界列（bit 0 和 bit N-1 为1），
    // ^ c.col 用XOR合入CPU端已放置皇后的列占用
    // ~(L-2) = ~(0b...11111110) = bit0和bitN-1为1
    unsigned int col = ~(L - 2) ^ (unsigned int)c.col;

    // ld_mem/rd_mem：捕获32位移位时溢出的位
    // ld << 1 时 bit 31 丢失 → ld_mem 捕获
    // rd >> 1 时 bit 0 丢失  → rd_mem 捕获
    // 这使得32位掩码可以表示超过32行的对角线状态
    unsigned int ld_mem = 0;
    unsigned int rd_mem = 0;

    // --- 2. 协作加载 jkl_queens 到共享内存 ---
    // jkl_queens[row]：边界皇后j/k/l在第row行产生的对角线约束（位掩码）
    // 整个workgroup共享一份，避免每个线程独立从全局内存加载
    __shared__ unsigned int jkl_queens[N];
    if (lid < N)
        jkl_queens[lid] = jkl_queens_arr[blockIdx.x * N + lid];
    __syncthreads();

    // --- 3. 处理伪星座 ---
    // start_ijkl的高12位存start（回溯起始行）
    // start==69 是伪星座标记（padWithPseudo填充的对齐用假数据）
    int start = c.start_ijkl >> 20;
    if (start == 69) {
        results[gid] = 0;
        return;
    }

    // --- 4. 修正边界皇后k和l的对角线 ---
    // 皇后k在第一列(col 0)，其左对角线(\)从右下向左上延伸
    // 皇后l在最后一列(col N-1)，其右对角线(/)从左下向右上延伸
    // GPU回溯从start行开始，需要根据start调整k/l对角线的起始位置
    int k_val = (c.start_ijkl >> 5) & 31;
    int l_val = c.start_ijkl & 31;
    if (k_val >= start)
        ld &= ~(L >> (k_val - start));  // 移除k皇后在start行的左对角线位
    if (l_val != N - 1 && l_val >= start)
        rd &= ~(1u << (l_val - start)); // 移除l皇后在start行的右对角线位

    // --- 5. 初始化回溯 ---
    int row = start;
    unsigned long long solutions = 0;

    // 计算起始行的可用位置：NOR(所有约束)
    // jkl_queens[row] 是边界皇后在该行的对角线约束
    unsigned int free_val = ~(ld | rd | col | jkl_queens[row]);

    // 位扫描：queen = 最低可用位
    // (-free) & free 的原理：负数补码保留最低位的1，AND后得到最低位
    // 等价于 ffs() 但不需要内置函数，GPU上只需2条指令(NEG+AND)
    unsigned int queen = (-free_val) & free_val;

    // 共享内存栈：存储每行放置的皇后位，用于回溯时恢复
    // 每个线程独立一份（WG_SIZE * N 个uint）
    __shared__ unsigned int queens[WG_SIZE][N];
    queens[lid][start] = queen;

    // direction: 1=前进(刚放置皇后), 0=回溯(刚移除皇后)
    // 用于统一尾代码中的防重复访问掩码计算
    int direction = 0;

    // --- 6. 主回溯循环：Jeff Somers 位并行 DFS ---
    // 循环条件：row >= start（回溯到起始行之前则结束）
    while (row >= start) {
        if (free_val) {
            // --- 情况a: 当前行有可用位置 ---

            if (row == N - 2) {
                // --- 叶子节点：到达倒数第二行 ---
                // N-1行（最后一行）已有边界皇后j，N-2行放置后即构成完整解
                solutions++;
                direction = 0;  // 接下来要回溯
                row--;

                // 从栈中加载上一行的皇后，恢复棋盘状态
                queen = queens[lid][row];

                // 恢复ld：右移1位（撤销前进时的<<1），用ld_mem恢复溢出位
                // __funnelshift_r(ld, ld_mem, 1) = (ld >> 1) | (ld_mem << 31)
                // & ~queen：移除该皇后的左对角线
                ld = __funnelshift_r(ld, ld_mem, 1) & ~queen;

                // 恢复rd：左移1位（撤销前进时的>>1），用rd_mem恢复溢出位
                // __funnelshift_l(rd_mem, rd, 1) = (rd << 1) | (rd_mem >> 31)
                // 注意：funnelshift_l第一参数是低位(lo)，第二参数是高位(hi)
                // & ~queen：移除该皇后的右对角线
                rd = __funnelshift_l(rd_mem, rd, 1) & ~queen;

                // 恢复记忆变量（撤销前进时的移位）
                ld_mem >>= 1;
                rd_mem <<= 1;
            } else {
                // --- 前进：在当前行放置皇后，进入下一行 ---
                direction = 1;
                queen = (-free_val) & free_val;  // 取最低可用位

                // 压栈：保存当前行放置的皇后位
                queens[lid][row] = queen;
                row++;

                // 更新ld_mem：捕获ld的bit 31（<<1时会丢失的位）
                // __funnelshift_l(ld, ld_mem, 1) = (ld_mem << 1) | (ld >> 31)
                ld_mem = __funnelshift_l(ld, ld_mem, 1);

                // 更新rd_mem：捕获rd的bit 0（>>1时会丢失的位）
                // __funnelshift_r(rd_mem, rd, 1) = (rd_mem >> 1) | (rd << 31)
                rd_mem = __funnelshift_r(rd_mem, rd, 1);

                // 更新ld：加入皇后的左对角线，整体左移1位（下一行的对角线位置）
                ld = (ld | queen) << 1;

                // 更新rd：加入皇后的右对角线，整体右移1位（下一行的对角线位置）
                rd = (rd | queen) >> 1;
            }
        } else {
            // --- 情况b: 当前行无可用位置，回溯 ---
            direction = 0;
            row--;

            // 从栈中弹出上一行的皇后
            queen = queens[lid][row];

            // 恢复ld/rd/ld_mem/rd_mem（与叶子节点回溯逻辑相同）
            ld = __funnelshift_r(ld, ld_mem, 1) & ~queen;
            rd = __funnelshift_l(rd_mem, rd, 1) & ~queen;
            ld_mem >>= 1;
            rd_mem <<= 1;
        }

        // --- 统一尾代码 ---

        // 计算当前行的可用位置：NOR(边界约束 | 左对角线 | 右对角线 | 列占用)
        free_val = ~(jkl_queens[row] | ld | rd | col);

        // 防重复访问掩码：避免在同一行重复尝试已经试过的位置
        // 前进(direction=1): ~(queen + 1 - 1) = ~queen，排除刚放的皇后
        // 回溯(direction=0): ~(queen + 0 - 1) = ~(queen-1)，排除queen及其右侧所有位
        //   （右侧=更低bit=更早尝试过的位置，因为每次取最低位）
        free_val &= ~(queen + direction - 1);

        // XOR翻转列占用：
        // 前进时：col加入皇后的列（XOR设置该位）
        // 回溯时：col移除皇后的列（同一XOR操作，第二次翻转恢复原值）
        // 这是关键的免分支技巧，避免了if/else
        col ^= queen;
    }

    // --- 7. 写回结果 ---
    results[gid] = (long long)solutions;
}

// =============================================================================
// Host 端辅助函数
// =============================================================================

// 按jkl排序星座：相同jkl的星座分到同一workgroup，共享jkl_queens共享内存
static void sortByJkl(std::vector<Constellation> &consts) {
    std::sort(consts.begin(), consts.end(), [](const Constellation &a, const Constellation &b) {
        return getJkl(a.start_ijkl) < getJkl(b.start_ijkl);
    });
}

// 填充伪星座：在每个jkl组之间插入假数据，使每组星座数为wgSize的倍数
// 确保每个workgroup内所有线程处理相同jkl的星座
// 伪星座标记：start_ijkl = (69 << 20)，内核检测到start==69直接返回0
static void padWithPseudo(std::vector<Constellation> &consts, int n, int wgSize) {
    sortByJkl(consts);
    std::vector<Constellation> padded;
    int allOnes = (1 << n) - 1;
    int currentJkl = getJkl(consts[0].start_ijkl);
    for (auto &c : consts) {
        if (getJkl(c.start_ijkl) != currentJkl) {
            // jkl组切换：填充伪星座直到对齐wgSize
            while (padded.size() % wgSize != 0)
                padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
            currentJkl = getJkl(c.start_ijkl);
        }
        padded.push_back(c);
    }
    // 末尾对齐
    while (padded.size() % wgSize != 0)
        padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
    consts = std::move(padded);
}

// 宏：为每个N值生成一个switch case分支
// 1. 请求最大共享内存carveout（RTX 4090为102KB）
// 2. 启动模板化的内核：<<<numWg, WG_SIZE>>>
#define LAUNCH_N(n_val) \
    case n_val: { \
        cudaFuncSetAttribute(n_queens_kernel<n_val>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
        n_queens_kernel<n_val><<<numWg, WG_SIZE>>>(d_const, d_jkl, d_results); \
        break; \
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
    padWithPseudo(constellations, n, WG_SIZE);
    long long cnt = constellations.size();
    int numWg = cnt / WG_SIZE;

    print_with_time("Total constellations (padded): %lld, workgroups: %d\n", cnt, numWg);

    // Step 3: 预计算 jkl_queens_arr
    // 对每个workgroup，根据其首个星座的ijkl计算边界皇后j/k/l在每一行的对角线约束
    // jkl_queens[row] 是一个位掩码，标记该行被j/k/l对角线占用的列
    unsigned int L = 1u << (n - 1);
    std::vector<uint32_t> jkl_queens_arr(numWg * n, 0);
    for (int wg = 0; wg < numWg; wg++) {
        // 取workgroup中第一个星座的ijkl（同组所有线程共享）
        int ijkl = constellations[wg * WG_SIZE].start_ijkl & 0xFFFFF;
        int j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

        // 第一轮：计算j和k/l从棋盘底部向顶部延伸的对角线
        // rdiag: j的右对角线 + k的右对角线（从底向上）
        unsigned int rdiag = (L >> j) | (L >> (n - 1 - k));
        // ldiag: j的左对角线 + l的左对角线（从底向上）
        unsigned int ldiag = (L >> j) | (L >> l);
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + (n - 1 - row)] = (ldiag >> row) | (rdiag << row) | L | 1;
            // | L | 1: 设置两个边界列（bit 0 和 bit N-1），因为l和k皇后占据这两列

        // 第二轮：叠加k和l从顶部向底部延伸的对角线
        ldiag = L >> k;   // k的左对角线（从顶向下）
        rdiag = 1u << l;  // l的右对角线（从顶向下）
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + row] |= (ldiag << row) | (rdiag >> row);

        // 边界皇后所在行的特殊处理：
        // k行：第一列已有皇后，清除bit N-1（左边界）避免冲突
        jkl_queens_arr[wg * n + k] = ~L;
        // l行：最后一列已有皇后，清除bit 0（右边界）避免冲突
        jkl_queens_arr[wg * n + l] = ~1u;
    }

    // Step 4: 转换为GPU端紧凑格式
    std::vector<Constellation4> const_data(cnt);
    for (long long i = 0; i < cnt; i++) {
        const_data[i] = {constellations[i].ld, constellations[i].rd,
                         constellations[i].col, constellations[i].start_ijkl};
    }

    // Step 5: 分配GPU内存并传输数据
    Constellation4 *d_const;
    unsigned int *d_jkl;
    long long *d_results;
    cudaMalloc(&d_const, cnt * sizeof(Constellation4));
    cudaMalloc(&d_jkl, numWg * n * sizeof(unsigned int));
    cudaMalloc(&d_results, cnt * sizeof(long long));

    cudaMemcpy(d_const, const_data.data(), cnt * sizeof(Constellation4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jkl, jkl_queens_arr.data(), numWg * n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, cnt * sizeof(long long));

    // Step 6: 启动GPU内核
    gettimeofday(&t0, NULL);
    switch (n) {
        LAUNCH_N(6)  LAUNCH_N(7)  LAUNCH_N(8)  LAUNCH_N(9)  LAUNCH_N(10)
        LAUNCH_N(11) LAUNCH_N(12) LAUNCH_N(13) LAUNCH_N(14) LAUNCH_N(15)
        LAUNCH_N(16) LAUNCH_N(17) LAUNCH_N(18) LAUNCH_N(19) LAUNCH_N(20)
        LAUNCH_N(21) LAUNCH_N(22) LAUNCH_N(23) LAUNCH_N(24) LAUNCH_N(25)
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
