#include "constellation.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#define WG_SIZE 64

struct Constellation4 {
    int ld, rd, col, start_ijkl;
};

template <int N>
__global__ void __launch_bounds__(WG_SIZE, 24)
n_queens_kernel(
    const Constellation4 * __restrict__ constellations,
    const unsigned int * __restrict__ jkl_queens_arr,
    long long * __restrict__ results)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    Constellation4 c = constellations[gid];

    const unsigned int L = 1u << (N - 1);
    unsigned int ld = (unsigned int)c.ld;
    unsigned int rd = (unsigned int)c.rd;
    unsigned int col = ~(L - 2) ^ (unsigned int)c.col;
    unsigned int ld_mem = 0;
    unsigned int rd_mem = 0;

    __shared__ unsigned int jkl_queens[N];
    if (lid < N)
        jkl_queens[lid] = jkl_queens_arr[blockIdx.x * N + lid];
    __syncthreads();

    int start = c.start_ijkl >> 20;
    if (start == 69) {
        results[gid] = 0;
        return;
    }

    int k_val = (c.start_ijkl >> 5) & 31;
    int l_val = c.start_ijkl & 31;
    if (k_val >= start)
        ld &= ~(L >> (k_val - start));
    if (l_val != N - 1 && l_val >= start)
        rd &= ~(1u << (l_val - start));

    int row = start;
    unsigned long long solutions = 0;

    unsigned int free_val = ~(ld | rd | col | jkl_queens[row]);
    unsigned int queen = (-free_val) & free_val;

    __shared__ unsigned int queens[WG_SIZE][N];
    queens[lid][start] = queen;

    int direction = 0;

    while (row >= start) {
        if (free_val) {
            if (row == N - 2) {
                solutions++;
                direction = 0;
                row--;
                queen = queens[lid][row];
                ld = __funnelshift_r(ld, ld_mem, 1) & ~queen;
                rd = __funnelshift_l(rd_mem, rd, 1) & ~queen;
                ld_mem >>= 1;
                rd_mem <<= 1;
            } else {
                direction = 1;
                queen = (-free_val) & free_val;
                queens[lid][row] = queen;
                row++;
                ld_mem = __funnelshift_l(ld, ld_mem, 1);
                rd_mem = __funnelshift_r(rd_mem, rd, 1);
                ld = (ld | queen) << 1;
                rd = (rd | queen) >> 1;
            }
        } else {
            direction = 0;
            row--;
            queen = queens[lid][row];
            ld = __funnelshift_r(ld, ld_mem, 1) & ~queen;
            rd = __funnelshift_l(rd_mem, rd, 1) & ~queen;
            ld_mem >>= 1;
            rd_mem <<= 1;
        }
        free_val = ~(jkl_queens[row] | ld | rd | col);
        free_val &= ~(queen + direction - 1);
        col ^= queen;
    }
    results[gid] = (long long)solutions;
}

static void sortByJkl(std::vector<Constellation> &consts) {
    std::sort(consts.begin(), consts.end(), [](const Constellation &a, const Constellation &b) {
        return getJkl(a.start_ijkl) < getJkl(b.start_ijkl);
    });
}

static void padWithPseudo(std::vector<Constellation> &consts, int n, int wgSize) {
    sortByJkl(consts);
    std::vector<Constellation> padded;
    int allOnes = (1 << n) - 1;
    int currentJkl = getJkl(consts[0].start_ijkl);
    for (auto &c : consts) {
        if (getJkl(c.start_ijkl) != currentJkl) {
            while (padded.size() % wgSize != 0)
                padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
            currentJkl = getJkl(c.start_ijkl);
        }
        padded.push_back(c);
    }
    while (padded.size() % wgSize != 0)
        padded.push_back({allOnes, allOnes, allOnes, (69 << 20), -2});
    consts = std::move(padded);
}

#define LAUNCH_N(n_val) \
    case n_val: { \
        cudaFuncSetAttribute(n_queens_kernel<n_val>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
        n_queens_kernel<n_val><<<numWg, WG_SIZE>>>(d_const, d_jkl, d_results); \
        break; \
    }

long long cuda_solve(int n, int preset_queens) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    auto constellations = generate_constellations(n, preset_queens);
    gettimeofday(&t1, NULL);
    print_with_time("Constellation generation: %.2fms, %zu subproblems\n",
                    time_diff_ms(t0, t1), constellations.size());

    if (constellations.empty()) {
        print_with_time("No subproblems generated!\n");
        return 0;
    }

    padWithPseudo(constellations, n, WG_SIZE);
    long long cnt = constellations.size();
    int numWg = cnt / WG_SIZE;

    print_with_time("Total constellations (padded): %lld, workgroups: %d\n", cnt, numWg);

    unsigned int L = 1u << (n - 1);
    std::vector<uint32_t> jkl_queens_arr(numWg * n, 0);
    for (int wg = 0; wg < numWg; wg++) {
        int ijkl = constellations[wg * WG_SIZE].start_ijkl & 0xFFFFF;
        int j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);
        unsigned int rdiag = (L >> j) | (L >> (n - 1 - k));
        unsigned int ldiag = (L >> j) | (L >> l);
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + (n - 1 - row)] = (ldiag >> row) | (rdiag << row) | L | 1;
        ldiag = L >> k;
        rdiag = 1u << l;
        for (int row = 0; row < n; row++)
            jkl_queens_arr[wg * n + row] |= (ldiag << row) | (rdiag >> row);
        jkl_queens_arr[wg * n + k] = ~L;
        jkl_queens_arr[wg * n + l] = ~1u;
    }

    std::vector<Constellation4> const_data(cnt);
    for (long long i = 0; i < cnt; i++) {
        const_data[i] = {constellations[i].ld, constellations[i].rd,
                         constellations[i].col, constellations[i].start_ijkl};
    }

    Constellation4 *d_const;
    unsigned int *d_jkl;
    long long *d_results;
    cudaMalloc(&d_const, cnt * sizeof(Constellation4));
    cudaMalloc(&d_jkl, numWg * n * sizeof(unsigned int));
    cudaMalloc(&d_results, cnt * sizeof(long long));

    cudaMemcpy(d_const, const_data.data(), cnt * sizeof(Constellation4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jkl, jkl_queens_arr.data(), numWg * n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, cnt * sizeof(long long));

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

    std::vector<long long> results(cnt);
    cudaMemcpy(results.data(), d_results, cnt * sizeof(long long), cudaMemcpyDeviceToHost);

    cudaFree(d_const);
    cudaFree(d_jkl);
    cudaFree(d_results);

    long long sum = 0;
    for (long long i = 0; i < cnt; i++) {
        if (constellations[i].start_ijkl >> 20 == 69)
            continue;
        int ijkl = constellations[i].start_ijkl & 0xFFFFF;
        sum += results[i] * symmetry(n, ijkl);
    }

    return sum;
}
