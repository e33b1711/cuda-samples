#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"

__device__ float db_abs(float d_signal) {
    return 20.0f * log10(d_signal);
}

__global__ void fill_bitmap_spec(uchar4 *ptr, int width, int height, float* d_signal, int color, bool clear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    if (clear){
        ptr[idx].x = 0;
        ptr[idx].y = 0;
        ptr[idx].z = 0;
        ptr[idx].w = 0;
    }


    const float scale = 2.0f;

    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = (x>0)? db_abs(d_signal[x-1]) : abs_x_mid;
    float abs_x_right = (x<width-1)? db_abs(d_signal[x+1]) : abs_x_mid;

    int y_mid = int( scale * abs_x_mid + height/2);
    int left_y = int( 0.5 * scale * (abs_x_left + abs_x_mid) + height/2);
    int right_y = int( 0.5 * scale * (abs_x_right + abs_x_mid) + height/2);
    int y_max = max(max(left_y, y_mid),right_y);
    int y_min = min(min(left_y, y_mid),right_y);

    if (y <= y_max and y >= y_min){
        if (color==0) ptr[idx].x = 255;
        if (color==1) ptr[idx].z = 255;
        if (color==2) ptr[idx].y = 255;
    }
}