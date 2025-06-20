#pragma once
#include <cuda_runtime.h>

__global__ void fill_bitmap_spec(uchar4 *ptr, int width, int height, int frame, float2* d_signal);