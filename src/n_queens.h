#pragma once

#include <vector>

using namespace std;

void n_queens(int N, int cur, int left, int right, long long &sum);

void n_queens_iterative(int N, int cur, int left, int right, long long &sum);

void partial_n_queens(int N, int cur, int left, int right, vector<int> &tot, int level);

void partial_n_queens_for_odd(int N, int cur, int left, int right, vector<int> &tot, int level);

long long parallel_n_queens(int N, int level);

long long cuda_n_queens(int N, int level);
