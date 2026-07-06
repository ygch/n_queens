// =============================================================================
// gpu_solver.h — GPU求解器接口
//
// 声明cuda_solve函数，供main.cpp调用。
// 实现在n_queens_cuda.cu中。
// =============================================================================

#pragma once

// CUDA D4星座法N-Queens求解器
// 参数：n — 棋盘大小，preset_queens — CPU预设皇后数
// 返回：N-Queens解的总数
long long cuda_solve(int n, int preset_queens);
