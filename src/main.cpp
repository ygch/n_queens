#include <stdio.h>
#include <stdlib.h>

#include "gpu_solver.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> [preset_queens]\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    if (N < 6 || N > 28) {
        printf("N must be in [6, 28], got %d\n", N);
        return -1;
    }

    int preset_queens = 6;
    if (argc >= 3)
        preset_queens = atoi(argv[2]);
    if (preset_queens < 4 || preset_queens >= N - 1) {
        printf("preset_queens must be in [4, N-2], got %d\n", preset_queens);
        return -1;
    }

    struct timeval start, end;

    print_with_time("===============================================================\n");
    gettimeofday(&start, NULL);
    long long sum = cuda_solve(N, preset_queens);
    gettimeofday(&end, NULL);
    print_with_time("cuda %d queens result %lld, calc time: [%.2fms]\n",
                    N, sum, time_diff_ms(start, end));
    print_with_time("===============================================================\n");

    return 0;
}
