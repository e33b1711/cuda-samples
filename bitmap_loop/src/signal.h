#pragma once
#include <cuda_runtime.h>

void generate_signal(float2* d_signal, const float phi, const int length, const int frame);
void run_fft(float2* d_signal, int length);

