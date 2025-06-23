#pragma once

#include <cuda_runtime.h>

void generate_signal(float2* d_signal, const float phi, const int length, const int frame);
