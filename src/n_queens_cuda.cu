#include <cuda_runtime.h>

#include "macro.h"
#include "n_queens.h"

inline int get_block_size(int size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens(int N, int *tot, long long *partial_sum, int cnt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        long long sum = 0;
        int last = (1 << N) - 1;
        int stack[192];
        int top = 0;

        stack[top++] = tot[tid * 3];
        stack[top++] = tot[tid * 3 + 1];
        stack[top++] = tot[tid * 3 + 2];

        while (top != 0) {
            int right = stack[--top];
            int left = stack[--top];
            int cur = stack[--top];

            if (cur == last) {
                sum++;
                continue;
            }

            int valid_pos = last & (~(cur | left | right));
            while (valid_pos) {
                int p = valid_pos & (-valid_pos);
                valid_pos -= p;
                stack[top++] = cur | p;
                stack[top++] = (left | p) << 1;
                stack[top++] = (right | p) >> 1;
            }
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

    int cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);

    int *cuda_tot;
    long long *cuda_partial_sum;
    CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * cnt * 3));
    CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * cnt));

    CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data(), sizeof(int) * cnt * 3, cudaMemcpyHostToDevice));
    CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * cnt));

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(get_block_size(cnt, CU1DBLOCK));

    printf("total size %d, block size %d, grid size %d\n", cnt, CU1DBLOCK, get_block_size(cnt, CU1DBLOCK));

    n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(err));
    }

    CU_SAFE_CALL(cudaMemcpy(partial_sum.data(), cuda_partial_sum, sizeof(long long) * cnt, cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    CU_SAFE_CALL(cudaFree(cuda_tot));
    CU_SAFE_CALL(cudaFree(cuda_partial_sum));

    return sum;
}
