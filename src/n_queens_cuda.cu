#include <cuda_runtime.h>
#include <omp.h>

#include "macro.h"
#include "n_queens.h"
#include "utils.h"

__device__ unsigned long long global_counter = 0;
__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt) {
    unsigned long long tid = atomicAdd(&global_counter, 1);
    const int last = (1 << N) - 1;

    while (tid < cnt) {
        long long sum = 0;
        int bottom = ((threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32) << 4;
        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if(valid_pos == 0) {
            tid = atomicAdd(&global_counter, 1);
            continue;
        }

        asm(".reg .s32 top, tmp, tmp2;\n\t"
            ".reg .s64 ltmp;\n\t"
            ".reg .pred p, q, z;\n\t"
            ".shared .align 16 .b8 stack[49152];\n\t"

            " mov.u32 top, %5;\n\t"
            " mov.u32 tmp, stack;\n\t"
            " add.s32 %5, %5, tmp;\n\t"
            " add.s32 top, top, tmp;\n\t"

            " st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"                // stack[top] = {cur, left, right, valid_pos}
            " add.s32 top, top, 512;\n\t"                                   // top += 512

            " LOOP:\n\t"
            " setp.eq.s32 p, top, %5;\n\t"                                  // top == bottom
            " @p bra FINISH;\n\t"                                           // done

            " ld.shared.v4.u32 {%1, %2, %3, %4}, [top + -512];\n\t"         // {cur, left, right, valid_pos} = stack[top - 512]
            " neg.s32 tmp, %4;\n\t"                                         // p = -valid_pos
            " and.b32 tmp, %4, tmp;\n\t"                                    // p = valid_pos & (-valid_pos)
            " sub.s32 %4, %4, tmp;\n\t"                                     // valid_pos -= p
            " st.shared.s32 [top + -500], %4;\n\t"                          // stack[top - 500] = valid_pos
            " setp.eq.s32 p, %4, 0;\n\t"                                    // p = (valid_pos == 0)
            " selp.b32 tmp2, 512, 0, p;\n\t"                                // tmp = (p == 1 ? 512 : 0)
            " sub.s32 top, top, tmp2;\n\t"                                  // top -= 512

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
            " selp.b64 ltmp, ltmp, 0, z;\n\t"                               // ltmp = (z == 1 ? ltmp : 0)
            " add.s64 %0, %0, ltmp;\n\t"                                    // sum += popc(valid_pos)

            " @!z st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"            // stack[top] = {cur, left, right, valid_pos}
            " selp.b32 tmp, 0, 512, z;\n\t"                                 // tmp = (z == 1 ? 0 : 512)
            " add.s32 top, top, tmp;\n\t"                                   // top += 512
            " bra.uni LOOP;\n\t"

            " FINISH:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos), "+r"(bottom)  // output
            :"r"(last), "r"(N - 1)                                          // input
        );

        partial_sum[tid] = sum;
        tid = atomicAdd(&global_counter, 1);
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

    // 2. divide total subproblems to different gpus（按 mod 交错分配）
    // 子问题 i 归卡 (i % gpu_num)：各卡从同一代价分布抽样，平衡的是期望负载，
    // 避免整块划分下负载方差随子问题代价聚集而放大（旧的比例表正是对该偏差的人工补偿）。
    // new_cnt/start_pos 同时描述各卡在主存 partial_sum 中的写回区域。
    vector<long long> new_cnt(gpu_num), start_pos(gpu_num);
    long long base = cnt / gpu_num, rem = cnt % gpu_num, acc = 0;
    for (int i = 0; i < gpu_num; i++) {
        new_cnt[i] = base + (i < rem ? 1 : 0);   // == #{ j | j % gpu_num == i }
        start_pos[i] = acc;
        acc += new_cnt[i];
    }

    // 3. use different gpu to process each trunk
#pragma omp parallel num_threads(gpu_num)
    {
        int idx = omp_get_thread_num();
        CU_SAFE_CALL(cudaSetDevice(idx));

        long long my_cnt = new_cnt[idx];
        print_with_time("gpu [%d] start job, with %lld(%.2f) subproblems.\n", idx, my_cnt,
                        my_cnt * 1.0 / cnt);

        // mod 分配：本卡负责子问题 { i | i % gpu_num == idx }，收集为连续段
        vector<int> my_tot;
        my_tot.reserve(my_cnt * 3);
        for (long long i = idx; i < cnt; i += gpu_num) {
            my_tot.push_back(tot[i * 3]);
            my_tot.push_back(tot[i * 3 + 1]);
            my_tot.push_back(tot[i * 3 + 2]);
        }

        int *cuda_tot;
        CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * my_cnt * 3));
        CU_SAFE_CALL(cudaMemcpy(cuda_tot, my_tot.data(), sizeof(int) * my_cnt * 3, cudaMemcpyHostToDevice));
        vector<int>().swap(my_tot);   // 及时释放主机内存

        long long *cuda_partial_sum;
        CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * my_cnt));
        CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * my_cnt));

        dim3 dimBlock(CU1DBLOCK);
        dim3 dimGrid(1024);

        n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, my_cnt);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel error: %s\n", cudaGetErrorString(err));
        }

        // 写回：区域 [start_pos, start_pos+my_cnt) 与其他卡互不重叠；
        // 段内结果对应 mod 序子问题，最终求和对所有子问题权重恒为 ×2，与顺序无关
        CU_SAFE_CALL(cudaMemcpy(partial_sum.data() + start_pos[idx], cuda_partial_sum, sizeof(long long) * my_cnt, cudaMemcpyDeviceToHost));

        CU_SAFE_CALL(cudaFree(cuda_tot));
        CU_SAFE_CALL(cudaFree(cuda_partial_sum));
        print_with_time("gpu [%d] finish job.\n", idx);
    }

    for (long long i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    return sum;
}
