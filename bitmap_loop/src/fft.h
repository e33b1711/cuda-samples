#pragma once
#include <cuda_runtime.h>

void run_fft(cudaStream_t stream, float2* t_domain, float2* f_domain, int length, int count);
void fft_postproc(cudaStream_t stream, float2* f_domain, uchar4* bitmap, const int block_len, const int n_blocks, int width, int height);


