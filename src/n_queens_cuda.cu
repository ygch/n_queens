#include <cuda_runtime.h>
#include <omp.h>

#include "macro.h"
#include "n_queens.h"
#include "utils.h"

inline int get_block_size(long long size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt) {
    const long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        int last = (1 << N) - 1;
        long long sum = 0;
        __shared__ int stack[48 * 256];
        const int bottom = threadIdx.x / 32 * 32 * STACKSIZE + threadIdx.x % 32;
        int top = bottom;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & (~(cur | left | right));

        if (valid_pos == 0) return;

        stack[top] = cur;
        stack[top + 32] = left;
        stack[top + 64] = right;
        stack[top + 96] = valid_pos;
        top += 128;

        while (top != bottom) {
            valid_pos = stack[top - 32];
            right = stack[top - 64];
            left = stack[top - 96];
            cur = stack[top - 128];

            int p = valid_pos & (-valid_pos);
            valid_pos -= p;
            stack[top - 32] = valid_pos;
            top -= (valid_pos == 0 ? 128 : 0);

            cur = cur | p;
            left = (left | p) << 1;
            right = (right | p) >> 1;
            valid_pos = last & (~(cur | left | right));

            if (valid_pos == 0) {
                continue;
            }

            p = cur ^ last;
            if((p & (p - 1)) == 0) {
                sum++; // only has one valid position
                continue;
            }

            stack[top] = cur;
            stack[top + 32] = left;
            stack[top + 64] = right;
            stack[top + 96] = valid_pos;
            top += 128;
        }

        partial_sum[tid] = sum;
    }
}

long long cuda_n_queens(int N, int rows) {
    struct timeval start, end;
    long long sum = 0;
    vector<int> tot;

    // 1. get total subproblems.
    gettimeofday(&start, NULL);
    partial_n_queens(N, 0, 0, 0, tot, rows);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, rows);
    }

    long long cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);
    gettimeofday(&end, NULL);

    print_with_time("Use %.2fms to generate %lld subproblems!\n", time_diff_ms(start, end), cnt);

    int gpu_num = 0;
    cudaGetDeviceCount(&gpu_num);
    if (gpu_num == 0) {
        printf("Failed to find any gpu!\n");
        return -1;
    }

    // 2. divide total subproblems to different trunks
    vector<long long> new_cnt(gpu_num), start_pos(gpu_num);
    long long total = 0;
    if (gpu_num == 8) {
        float ratio[8] = {0.18, 0.16, 0.13, 0.12, 0.11, 0.1, 0.1, 0.1};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else if (gpu_num == 4) {
        float ratio[4] = {0.34, 0.25, 0.21, 0.2};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else if (gpu_num == 2) {
        float ratio[2] = {0.59, 0.41};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else {
        long long partial_cnt = cnt / gpu_num;
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = partial_cnt;
            start_pos[i] = total;
            total += partial_cnt;
        }
    }
    new_cnt[gpu_num - 1] = cnt - total;
    start_pos[gpu_num - 1] = total;

    // 3. use different gpu to process each trunk
#pragma omp parallel num_threads(gpu_num)
    {
        int idx = omp_get_thread_num();
        CU_SAFE_CALL(cudaSetDevice(idx));

        long long cnt = new_cnt[idx];

        print_with_time("gpu [%d] start job, with %lld subproblems.\n", idx, cnt);

        int *cuda_tot;
        CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * cnt * 3));
        CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data() + start_pos[idx] * 3, sizeof(int) * cnt * 3, cudaMemcpyHostToDevice));

        long long *cuda_partial_sum;
        CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * cnt));
        CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * cnt));

        dim3 dimBlock(CU1DBLOCK);
        dim3 dimGrid(get_block_size(cnt, CU1DBLOCK));

        n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel error: %s\n", cudaGetErrorString(err));
        }

        CU_SAFE_CALL(cudaMemcpy(partial_sum.data() + start_pos[idx], cuda_partial_sum, sizeof(long long) * cnt, cudaMemcpyDeviceToHost));

        CU_SAFE_CALL(cudaFree(cuda_tot));
        CU_SAFE_CALL(cudaFree(cuda_partial_sum));
        print_with_time("gpu [%d] finish job.\n", idx);
    }

    for (long long i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    return sum;
}
