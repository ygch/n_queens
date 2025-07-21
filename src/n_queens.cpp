#include "n_queens.h"

#include <omp.h>
#include <stdlib.h>

#include "utils.h"

void n_queens(int N, int cur, int left, int right, long long &sum) {
    int last = (1 << N) - 1;
    if (cur == last) {
        sum++;
        return;
    }

    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        n_queens(N, cur | p, (left | p) << 1, (right | p) >> 1, sum);
    }
}

// a little slower
void n_queens_iterative(int N, int cur, int left, int right, long long &sum) {
    int last = (1 << N) - 1;
    int stack[64];
    int top = 0;
    int valid_pos = last & (~(cur | left | right));

    if (valid_pos == 0) return;

    stack[top++] = cur;
    stack[top++] = left;
    stack[top++] = right;
    stack[top++] = valid_pos;

    while (top != 0) {
        valid_pos = stack[top - 1];
        right = stack[top - 2];
        left = stack[top - 3];
        cur = stack[top - 4];

        int p = valid_pos & (-valid_pos);
        valid_pos -= p;

        if (valid_pos == 0) {
            top -= 4;
        } else {
            stack[top - 1] = valid_pos;
        }

        cur = cur | p;
        left = (left | p) << 1;
        right = (right | p) >> 1;
        valid_pos = last & (~(cur | left | right));

        if (valid_pos == 0) {
            continue;
        }

        p = cur ^ last;
        if((p & (p - 1)) == 0) {
            sum += __builtin_popcount(valid_pos);
            continue;
        }

        stack[top++] = cur;
        stack[top++] = left;
        stack[top++] = right;
        stack[top++] = valid_pos;
    }
}

void partial_n_queens(int N, int cur, int left, int right, vector<int> &tot, int rows) {
    int last = (1 << N) - 1;
    if (cur == 0) {
        last = (1 << N / 2) - 1;
    }
    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        if (rows == 1) {
            tot.push_back(cur | p);
            tot.push_back((left | p) << 1);
            tot.push_back((right | p) >> 1);
            continue;
        }
        partial_n_queens(N, cur | p, (left | p) << 1, (right | p) >> 1, tot, rows - 1);
    }
}

void partial_n_queens_for_odd(int N, int cur, int left, int right, vector<int> &tot, int rows) {
    int last = (1 << N) - 1;
    if (cur == 0) {
        last = (1 << N / 2);
    } else if ((cur & (cur - 1)) == 0) {
        last = (1 << (N - 2) / 2) - 1;
    }

    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        if (rows == 1) {
            tot.push_back(cur | p);
            tot.push_back((left | p) << 1);
            tot.push_back((right | p) >> 1);
            continue;
        }
        partial_n_queens_for_odd(N, cur | p, (left | p) << 1, (right | p) >> 1, tot, rows - 1);
    }
}

long long serial_n_queens(int N, int rows) {
    long long sum = 0;
    vector<int> tot;

    partial_n_queens(N, 0, 0, 0, tot, rows);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, rows);
    }

    int cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);

    for (int i = 0; i < cnt; i++) {
        n_queens(N, tot[3 * i], tot[3 * i + 1], tot[3 * i + 2], partial_sum[i]);
    }

    for (int i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    return sum;
}

long long parallel_n_queens(int N, int rows) {
    long long sum = 0;
    vector<int> tot;

    partial_n_queens(N, 0, 0, 0, tot, rows);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, rows);
    }

    int cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);

    omp_set_num_threads(32);
#pragma omp parallel for
    for (int i = 0; i < cnt; i++) {
        n_queens(N, tot[3 * i], tot[3 * i + 1], tot[3 * i + 2], partial_sum[i]);
    }

    for (int i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    return sum;
}
