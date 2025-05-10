#include <cuda_runtime.h>

#include "macro.h"
#include "n_queens.h"

inline int get_block_size(int size, int block_size) { return (size + block_size - 1) / block_size; }

__device__ void n_queens_device(int N, int cur, int left, int right, long long *sum) {
    int last = (1 << N) - 1;
    if (cur == last) {
        (*sum)++;
        return;
    }

    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        n_queens_device(N, cur | p, (left | p) << 1, (right | p) >> 1, sum);
    }
}

__device__ long long n_queens_device_iterative(int N, int cur, int left, int right) {
    long long sum = 0;
    int last = (1 << N) - 1;
    int stack[192];
    int top = 0;
    stack[top++] = cur;
    stack[top++] = left;
    stack[top++] = right;

    while (top != 0) {
        right = stack[--top];
        left = stack[--top];
        cur = stack[--top];

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

    return sum;
}

__global__ void n_queens(int N, int *tot, long long *partial_sum, int cnt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        partial_sum[tid] = n_queens_device_iterative(N, tot[tid * 3], tot[tid * 3 + 1], tot[tid * 3 + 2]);
    }
}

long long cuda_n_queens(int N, int level) {
    long long sum = 0;
    vector<int> tot;

    partial_n_queens(N, 0, 0, 0, tot, 0, level);

    int cnt = tot.size() / 3;

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, 0, level);
    }

    int new_cnt = tot.size() / 3;
    vector<long long> partial_sum(new_cnt);

    // random_shuffle(tot.data(), cnt);
    // random_shuffle(tot.data() + cnt * 3, new_cnt - cnt);

    int *cuda_tot;
    long long *cuda_partial_sum;
    CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * new_cnt * 3));
    CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * new_cnt));

    CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data(), sizeof(int) * new_cnt * 3, cudaMemcpyHostToDevice));
    CU_SAFE_CALL(cudaMemcpy(cuda_partial_sum, partial_sum.data(), sizeof(long long) * new_cnt, cudaMemcpyHostToDevice));

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(get_block_size(new_cnt, CU1DBLOCK));

    printf("total size %d, block size %d, grid size %d\n", new_cnt, CU1DBLOCK, get_block_size(new_cnt, CU1DBLOCK));

    n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, new_cnt);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(err));
    }

    CU_SAFE_CALL(cudaMemcpy(partial_sum.data(), cuda_partial_sum, sizeof(long long) * new_cnt, cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    for (int i = cnt; i < new_cnt; i++) {
        sum += partial_sum[i];
    }

    CU_SAFE_CALL(cudaFree(cuda_tot));
    CU_SAFE_CALL(cudaFree(cuda_partial_sum));

    return sum;
}
