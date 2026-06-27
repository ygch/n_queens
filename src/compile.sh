#!/bin/sh

set -x

# OpenCL build for the D4 constellation solver (ported from NQueensFAF)
# RTX 4090
g++ -O3 -std=c++17 \
    -DCL_TARGET_OPENCL_VERSION=300 \
    -I /usr/local/cuda/include \
    -L /usr/local/cuda/lib64 \
    main.cpp constellation.cpp gpu_solver.cpp utils.cpp \
    -o n_queens -lOpenCL
