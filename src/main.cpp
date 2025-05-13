#include <stdio.h>
#include <stdlib.h>

#include "n_queens.h"

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    if(N < 2 || N > 22) {
        printf("Not allowed %d queens problem!\n", N);
        return -1;
    }

    int level = atoi(argv[2]);
    if(level<2) {
        printf("level shold be at least 2, so set to 2!\n");
        level = 2;
    }

    if(level > 9) {
        printf("level should be smaller than min(%d, 9)\n", N);
        level = min(N, 9);
    }
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
