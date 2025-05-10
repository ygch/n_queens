#!/bin/sh

set -x

nvcc -gencode arch=compute_89,code=sm_89 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart n_queens_cuda.cu n_queens.cpp main.cpp -o n_queens -O3 -Xcompiler -fopenmp
