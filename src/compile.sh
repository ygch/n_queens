#!/bin/sh

# =============================================================================
# compile.sh — CUDA D4星座法N-Queens求解器编译脚本
#
# 编译目标：RTX 4090 (sm_89, Ada Lovelace架构)
# 需要安装CUDA Toolkit (nvcc)
#
# 编译选项说明：
#   -O3                    最高优化级别
#   -std=c++17             C++17标准
#   -gencode arch=compute_89,code=sm_89   目标GPU架构(RTX 4090)
#   --ptxas-options=-v     输出寄存器/共享内存统计（验证优化用）
#   -allow-unsupported-compiler  兼容较新版本的gcc/nvcc组合
#   -Xcompiler -fopenmp    传递给host编译器，启用OpenMP
#
# 源文件：
#   n_queens_cuda.cu  — CUDA内核 + host启动函数（核心）
#   constellation.cpp — D4对称星座生成
#   utils.cpp         — 工具函数（计时、带时间戳输出）
#   main.cpp          — 命令行入口
# =============================================================================

set -x

nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 \
    --ptxas-options=-v -allow-unsupported-compiler \
    n_queens_cuda.cu constellation.cpp utils.cpp main.cpp \
    -o n_queens -Xcompiler -fopenmp
