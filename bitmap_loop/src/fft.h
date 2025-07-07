#pragma once
#include <cuda_runtime.h>

#include "params.h"

void run_fft(cudaStream_t stream, float2 *t_domain, float2 *f_domain, const ps params, bool clear);
void fft_postproc(cudaStream_t stream, float2 *f_domain, uchar4 *bitmap, const ps params, bool clear);
