#pragma once
#include <cuda_runtime.h>

void polchrome(cudaStream_t stream, float2 *f_domain, uchar4 *bitmap, const int block_len, const int n_blocks, const int width);