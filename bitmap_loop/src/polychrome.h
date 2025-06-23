#pragma once
#include <cuda_runtime.h>

void polchrome(float2* f_domain, unsigned uchar4* hist, const int n_bins, const int n_spec);