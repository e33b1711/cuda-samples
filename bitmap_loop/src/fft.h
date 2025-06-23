#pragma once
#include <cuda_runtime.h>

void run_fft(float2* t_domain, float2* f_domain, int length, int count);
void fft_postproc(float2* f_domain, uchar4* bitmap, const int length, const int count);


