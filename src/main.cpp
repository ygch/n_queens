#include <stdio.h>
#include <stdlib.h>

#include "n_queens.h"

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    int level = atoi(argv[2]);
    long long sum = 0;

    struct timeval start, end;

    /*
    gettimeofday(&start, NULL);
    n_queens(N, 0, 0, 0, sum);
    gettimeofday(&end, NULL);
    printf("%d queens result %lld, calc time: [%.2fms]\n", N, sum,
           time_diff_ms(start, end));
    */
    gettimeofday(&start, NULL);
    sum = parallel_n_queens(N, level);
    gettimeofday(&end, NULL);
    printf("parallel %d queens result %lld, calc time: [%.2fms]\n", N, sum,
           time_diff_ms(start, end));

    gettimeofday(&start, NULL);
    sum = cuda_n_queens(N, level);
    gettimeofday(&end, NULL);
    printf("cuda %d queens result %lld, calc time: [%.2fms]\n", N, sum, time_diff_ms(start, end));

    return 0;
}
