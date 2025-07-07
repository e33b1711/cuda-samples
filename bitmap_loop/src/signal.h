#pragma once

#include <cuda_runtime.h>

void generate_signal(float2 *d_signal, const int length);

void inject_spike(cudaStream_t stream, float2* f_domain, const int block_len, const int n_blocks, float2 value, int repetetions, int index);
