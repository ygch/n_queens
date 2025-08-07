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
        const int bottom = (threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32;
        int top = bottom;

        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        asm(".reg .s32 base_addr, top, tmp, tmp2;\n\t"
            ".reg .s64 ltmp, ltmp2;\n\t"
            ".reg .pred p, q, z;\n\t"
            ".shared .align 16 .b8 stack[49152];\n\t"

            " mov.u32 top, %5;\n\t"
            " mov.u32 base_addr, stack;\n\t"
            " mad.lo.s32 tmp2, top, 16, base_addr;\n\t"
            " st.shared.v4.u32 [tmp2], {%1, %2, %3, %4};\n\t"               // stack[top] = {cur, left, right, valid_pos}
            " setp.eq.s32 p, %4, 0;\n\t"                                    // valid_pos == 0
            " @p bra FINISH;\n\t"                                           // return;
            " add.s32 top, top, 32;\n\t"                                    // top += 32

            " LOOP:\n\t"
            " setp.eq.s32 p, top, %6;\n\t"                                  // top == bottom
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
            " and.b32 %4, %7, tmp;\n\t"                                     // valid_pos = last & tmp
            " popc.b32 tmp, %1;\n\t"                                        // tmp = popc(cur)
            " setp.eq.s32 p, tmp, %8;\n\t"                                  // popc(cur) == N - 1
            " setp.eq.s32 q, %4, 0;\n\t"                                    // valid_pos == 0
            " or.pred z, p, q;\n\t"                                         // valid_pos == 0 || popc(cur) == N - 1

            " popc.b32 tmp, %4;\n\t"                                        // tmp = popc(valid_pos)
            " cvt.s64.s32 ltmp, tmp;\n\t"
            " cvt.s64.s32 ltmp2, 0;\n\t"
            " selp.s64 ltmp, ltmp, ltmp2, z;\n\t"                           // ltmp = (z == 1 ? ltmp : ltmp2)
            " add.s64 %0, %0, ltmp;\n\t"                                    // sum += 1

            " @!z mad.lo.s32 tmp2, top, 16, base_addr;\n\t"
            " @!z st.shared.v4.u32 [tmp2], {%1, %2, %3, %4};\n\t"           // stack[top] = {cur, left, right, valid_pos}
            " @!z add.s32 top, top, 32;\n\t"                                // top += 32
            " bra.uni LOOP;\n\t"

            " FINISH:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos) // output
            :"r"(top), "r"(bottom), "r"(last), "r"(N - 1)                   // input
        );

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
        //float ratio[8] = {0.198, 0.156, 0.128, 0.117, 0.105, 0.100, 0.098, 0.098};
        float ratio[8] = {0.18, 0.16, 0.13, 0.12, 0.11, 0.100, 0.10, 0.10};
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
