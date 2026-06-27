#!/bin/sh

set -x

# CUDA build for the D4 constellation solver (ported from NQueensFAF)
# RTX 4090 = sm_89
nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 \
    --ptxas-options=-v -allow-unsupported-compiler \
    n_queens_cuda.cu constellation.cpp utils.cpp main.cpp \
    -o n_queens -Xcompiler -fopenmp
