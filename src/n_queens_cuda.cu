#include <cuda_runtime.h>

#include "macro.h"
#include "n_queens.h"

inline int get_block_size(long long size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        int last = (1 << N) - 1;
        long long sum = 0;
        __shared__ int stack[48 * 256];
        int idx = threadIdx.x / 32 * 32 * 96 + threadIdx.x % 32;
        int top = 0;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & (~(cur | left | right));

        if(valid_pos == 0) return;

        stack[idx + top] = cur;
        stack[idx + top + 32] = left;
        stack[idx + top + 64] = right;
        stack[idx + top + 96] = valid_pos;
        top += 128;

        while (top != 0) {
            valid_pos = stack[idx + top - 32];
            right = stack[idx + top - 64];
            left = stack[idx + top - 96];
            cur = stack[idx + top - 128];

            int p = valid_pos & (-valid_pos);
            valid_pos -= p;

            if(valid_pos == 0) {
                top -= 128;
            } else {
                stack[idx + top - 32] = valid_pos;
            }

            cur = cur | p;
            if(cur == last) {
                sum++;
                continue;
            }

            left = (left | p) << 1;
            right = (right | p) >> 1;
            valid_pos = last & (~(cur | left | right));

            if(valid_pos == 0) {
                continue;
            }

            stack[idx + top] = cur;
            stack[idx + top + 32] = left;
            stack[idx + top + 64] = right;
            stack[idx + top + 96] = valid_pos;
            top += 128;
        }

        partial_sum[tid] = sum;
    }
}

long long cuda_n_queens(int N, int level) {
    long long sum = 0;
    vector<int> tot;

    partial_n_queens(N, 0, 0, 0, tot, level);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, level);
    }

    long long cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    int *cuda_tot;
    long long *cuda_partial_sum;
    CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * cnt * 3));
    CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * cnt));

    CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data(), sizeof(int) * cnt * 3, cudaMemcpyHostToDevice));
    CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * cnt));

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(get_block_size(cnt, CU1DBLOCK));

    printf("total size %lld, block size %d, grid size %d\n", cnt, CU1DBLOCK, get_block_size(cnt, CU1DBLOCK));

    n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(err));
    }

    CU_SAFE_CALL(cudaMemcpy(partial_sum.data(), cuda_partial_sum, sizeof(long long) * cnt, cudaMemcpyDeviceToHost));

    for (long long i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    CU_SAFE_CALL(cudaFree(cuda_tot));
    CU_SAFE_CALL(cudaFree(cuda_partial_sum));

    return sum;
}
