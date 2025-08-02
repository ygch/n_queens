#include <cuda_runtime.h>

#include "macro.h"
#include "n_queens.h"

inline int get_block_size(long long size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens_global(int N, int *tot, long long *partial_sum, long long cnt) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        int last = (1 << N) - 1;
        long long sum = 0;
        extern __shared__ int4 stack[];
        int idx = threadIdx.x / 32 * 32 * 24 + threadIdx.x % 32;
        int top = idx;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~(cur | left | right);

        if(valid_pos == 0) return;

        stack[top] = make_int4(cur, left, right, valid_pos);
        top += 32;

        while (top != idx) {
            valid_pos = stack[top - 32].w;
            right = stack[top - 32].z;
            left = stack[top - 32].y;
            cur = stack[top - 32].x;

            int p = valid_pos & (-valid_pos);
            valid_pos -= p;
            stack[top - 32].w = valid_pos;
            top -= (valid_pos == 0 ? 32 : 0);

            cur = cur | p;
            left = (left | p) << 1;
            right = (right | p) >> 1;
            valid_pos = last & ~(cur | left | right);

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

long long cuda_n_queens(int N, int level) {
    long long sum = 0;
    vector<int> tot;

    partial_n_queens(N, 0, 0, 0, tot, level);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, level);
    }

    long long cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);
    
    printf("total size %lld, block size %d, grid size %d\n", cnt, CU1DBLOCK, get_block_size(cnt, CU1DBLOCK));

    int *cuda_tot;
    long long *cuda_partial_sum;
    CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * cnt * 3));
    CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * cnt));

    CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data(), sizeof(int) * cnt * 3, cudaMemcpyHostToDevice));
    CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * cnt));

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(get_block_size(cnt, CU1DBLOCK));    
    
    int shared_memory_size = 96 * 1024;
    cudaFuncSetAttribute((void*)n_queens_global, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

    n_queens_global<<<dimGrid, dimBlock, shared_memory_size>>>(N, cuda_tot, cuda_partial_sum, cnt);

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
