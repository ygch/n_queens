#include <stdio.h>
#include <stdlib.h>

#include "n_queens.h"
#include "macro.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    if (N < 2 || N > 28) {
        printf("Not allowed %d queens problem!\n", N);
        return -1;
    }

    int rows = atoi(argv[2]);
    if(STACKSIZE + 1 + rows < N) {
        printf("This configure cannot process %d-queens with pre-placing %d rows. Pre-placed rows should be >= %d.\n", N, rows, N - STACKSIZE - 1);
        return -1;
    }

    if (rows < 2) {
        printf("Pre-placed rows shold be at least 2, so set to 2!\n");
        rows = 2;
    }

    long long sum = 0;

    struct timeval start, end;

    /*
    gettimeofday(&start, NULL);
    sum = serial_n_queens(N, rows);
    gettimeofday(&end, NULL);
    printf("serial %d queens result %lld, calc time: [%.2fms]\n", N, sum,
           time_diff_ms(start, end));

    gettimeofday(&start, NULL);
    sum = parallel_n_queens(N, rows);
    gettimeofday(&end, NULL);
    printf("parallel %d queens result %lld, calc time: [%.2fms]\n", N, sum,
           time_diff_ms(start, end));
    */

    print_with_time("===============================================================\n");
    gettimeofday(&start, NULL);
    sum = cuda_n_queens(N, rows);
    gettimeofday(&end, NULL);
    print_with_time("cuda %d queens result %lld, calc time: [%.2fms]\n", N, sum, time_diff_ms(start, end));
    print_with_time("===============================================================\n");

    return 0;
}
