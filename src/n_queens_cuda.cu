#include <cuda_runtime.h>

#include "macro.h"
#include "n_queens.h"

inline int get_block_size(long long size, int block_size) { return (size + block_size - 1) / block_size; }

__global__ void n_queens_global(int N, int *tot, long long *partial_sum, long long cnt) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt) {
        int last = (1 << N) - 1;
        long long sum = 0;
        //extern __shared__ int4 stack[];
        int bottom = (threadIdx.x / 32) * 32 * 24 + threadIdx.x % 32;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & (~(cur | left | right));

        if(valid_pos == 0) return;

        asm(".reg .s32 base_addr, top, tmp, tmp2;\n\t"
            ".reg .s64 ltmp;\n\t"
            ".reg .pred p, q, z;\n\t"
            ".extern .shared .align 16 .b8 stack[];\n\t"

            " mov.u32 top, %5;\n\t"
            " mov.u32 base_addr, stack;\n\t"

            " mad.lo.s32 tmp2, top, 16, base_addr;\n\t"                     // calculate stack top address
            " st.shared.v4.u32 [tmp2], {%1, %2, %3, %4};\n\t"               // stack[top] = {cur, left, right, valid_pos}
            " add.s32 top, top, 32;\n\t"                                    // top += 32

            " LOOP:\n\t"
            " setp.eq.s32 p, top, %5;\n\t"                                  // top == bottom
            " @p bra FINISH;\n\t"                                           // done

            " mad.lo.s32 tmp2, top, 16, base_addr;\n\t"
            " ld.shared.v4.u32 {%1, %2, %3, %4}, [tmp2 + -512];\n\t"        // {cur, left, right, valid_pos} = stack[top - 32]

            " neg.s32 tmp, %4;\n\t"                                         // p = -valid_pos
            " and.b32 tmp, %4, tmp;\n\t"                                    // p = valid_pos & (-valid_pos)
            " sub.s32 %4, %4, tmp;\n\t"                                     // valid_pos -= p
            " st.shared.s32 [tmp2 + -500], %4;\n\t"                         // stack[top - 32] = valid_pos
            " setp.eq.s32 p, %4, 0;\n\t"                                    // p = (valid_pos == 0)
            " selp.b32 tmp2, 32, 0, p;\n\t"                                 // tmp2 = (p==1 ? 32 : 0)
            " sub.s32 top, top, tmp2;\n\t"                                  // top -= 32

            " or.b32 %1, %1, tmp;\n\t"                                      // cur = cur | p
            " or.b32 %2, %2, tmp;\n\t"                                      // left = left | p
            " shl.b32 %2, %2, 1;\n\t"                                       // left = left << 1
            " or.b32 %3, %3, tmp;\n\t"                                      // right = right | p
            " shr.b32 %3, %3, 1;\n\t"                                       // right = right >> 1
            " lop3.b32 tmp, %1, %2, %3, 0x1;\n\t"                           // tmp = ~cur & ~left & ~right;
            " and.b32 %4, %6, tmp;\n\t"                                     // valid_pos = last & tmp
            " popc.b32 tmp, %1;\n\t"                                        // tmp = popc(cur)
            " setp.eq.s32 p, tmp, %7;\n\t"                                  // popc(cur) == N - 1
            " setp.eq.s32 q, %4, 0;\n\t"                                    // valid_pos == 0
            " or.pred z, p, q;\n\t"                                         // valid_pos == 0 || popc(cur) == N - 1

            " popc.b32 tmp, %4;\n\t"                                        // tmp = popc(valid_pos)
            " cvt.s64.s32 ltmp, tmp;\n\t"                                   // s32 -> s64
            " @z add.s64 %0, %0, ltmp;\n\t"                                 // sum += popc(valid_pos)

            " mad.lo.s32 tmp2, top, 16, base_addr;\n\t"                     // remove @!z to reduce branches
            " @!z st.shared.v4.u32 [tmp2], {%1, %2, %3, %4};\n\t"           // stack[top] = {cur, left, right, valid_pos}
            " @!z add.s32 top, top, 32;\n\t"                                // top += 32
            " bra.uni LOOP;\n\t"

            " FINISH:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos) // output
            :"r"(bottom), "r"(last), "r"(N - 1)                             // input
        );

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
