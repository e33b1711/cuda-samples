#pragma once
#include <cuda_runtime.h>

void generate_signal(float2* d_signal, const float phi, const int length, const int frame);
void run_fft(float2* t_domain, float2* f_domain, int length, int count);
void fft_abs(float2* f_domain, float* f_abs, int length, int count);
void fft_postproc(float2* f_domain, float* f_max, float* f_min, float* f_mean, const int length, const int count);

