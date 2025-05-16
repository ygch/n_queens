// Copyright 2022 Welink Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the License); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
// Author: yaoguangchao@pyou.com

#pragma once

#include <stdio.h>

#define TIMEDIFF(s, e) ((e.tv_sec - s.tv_sec) * 1000.0 + (e.tv_usec - s.tv_usec) / 1000.0)

#define TRUNCATE(a) ((a) > 255.0 ? 255 : ((a) < 0 ? 0 : (rint((a)))))

// The size of a CUDA 1-d block, e.g. for vector operations..
#define CU1DBLOCK 192

// The size of edge of CUDA square block, e.g. for matrix operations.
#define CU2DBLOCK 16

#define CU_SAFE_CALL(fun)                                                                        \
    {                                                                                            \
        int ret;                                                                                 \
        if ((ret = (fun)) != 0) {                                                                \
            fprintf(stderr, "[%s:%s(%d)]cudaError_t %d:%s\n", __FILE__, __func__, __LINE__, ret, \
                    cudaGetErrorString((cudaError_t)ret));                                       \
            exit(-1);                                                                            \
        }                                                                                        \
        cudaDeviceSynchronize();                                                                 \
    }
