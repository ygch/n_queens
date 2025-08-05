#include <cuda_runtime.h>
#include <omp.h>

#include "macro.h"
#include "n_queens.h"
#include "utils.h"

inline int get_block_size(long long size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt) {
    const long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int last = (1 << N) - 1;

    if (tid < cnt) {
        long long sum = 0;
        __shared__ int4 stack[48 * 64];
        const int bottom = (threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32;
        int top = bottom;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if (valid_pos == 0) return;

        stack[top] = make_int4(cur, left, right, valid_pos);
        top += 32;

        while (top != bottom) {
            valid_pos = stack[top - 32].w;
            right = stack[top - 32].z;
            left = stack[top - 32].y;
            cur = stack[top - 32].x;

#if 1
            int p = valid_pos & (-valid_pos);
            valid_pos -= p;
            stack[top - 32].w = valid_pos;
            top -= (valid_pos == 0 ? 32 : 0);

            cur = cur | p;
            left = (left | p) << 1;
            right = (right | p) >> 1;
            valid_pos = last & ~cur & ~left & ~right;
#else
            asm(".reg .s32 t;\n\t"
                ".reg .pred p;\n\t"
                " neg.s32 t, %3;\n\t"               // t = -valid_pos
                " and.b32 t, %3, t;\n\t"            // t = valid_pos & (-valid_pos)
                " sub.s32 %3, %3, t;\n\t"           // valid_pos -= t
                " mov.s32 %4, %3;\n\t"              // stack[top - 32].w = valid_pos
                " setp.eq.s32 p, %3, 0;\n\t"        // valid_pos == 0
                " @p sub.s32 %5, %5, 32;\n\t"       // top -= 32
                " or.b32 %0, %0, t;\n\t"            // cur = cur | p
                " or.b32 %1, %1, t;\n\t"            // left = left | p
                " shl.b32 %1, %1, 1;\n\t"           // left = left << 1
                " or.b32 %2, %2, t;\n\t"            // right = right | p
                " shr.b32 %2, %2, 1;\n\t"           // right = right >> 1
                " lop3.b32 t, %0, %1, %2, 0x1;\n\t" // t = ~cur & ~left & ~right;
                " and.b32 %3, %6, t;\n\t"           // valid_pos = last & t
               :"+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos), "=r"(stack[top - 32].w), "+r"(top) // output
               : "r"(last)); // input
#endif
            if (valid_pos == 0 || __popc(cur) == N - 1) {
                sum += __popc(valid_pos);
                continue;
            }

            stack[top] = make_int4(cur, left, right, valid_pos);
            top += 32;
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
