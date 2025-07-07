#pragma once
#include <cuda_runtime.h>

void polchrome(cudaStream_t stream, float2 *f_domain, uchar4 *bitmap, const ps params);