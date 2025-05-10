#pragma once

#include <sys/time.h>

#include <vector>

using namespace std;

float time_diff_ms(struct timeval &start, struct timeval &end);

void n_queens(int N, int cur, int left, int right, long long &sum);

void partial_n_queens(int N, int cur, int left, int right, vector<int> &tot, int cur_level, int level);

void partial_n_queens_for_odd(int N, int cur, int left, int right, vector<int> &tot, int cur_level, int level);

void random_shuffle(int *data, int len);

long long parallel_n_queens(int N, int level);

long long cuda_n_queens(int N, int level);
