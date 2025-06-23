#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#include "aux.h"
#include "fft.h"

__device__ uchar4 mapping(unsigned short hist_count, const int n_spec ){

    assert(hist_count<=n_spec);

    //0 => 0 0 0 0
    if (hist_count==0) return uchar4(0,0,0,0);

    // 1 => 0.0 / 0
    // n_spec => 1.0 / 510
    float color_index = log(hist_count) / log(n_spec);
    int c_index = int(510.0 * color_index);

    //index: 0-255
    //       0-255
    //       255-0
    //       0
    if(c_index<256){
        return uchar(c_index, 255-c_index, 0, 0);
    }
    //index: 256-510
    //       254-0
    //       0
    //       1-255
    if(c_index<510){
        return uchar(510-c_index, 0, c_index-255, 0);
    }

    return uchar(255, 255, 255, 0);
}

__device__ float db_abs(float2 fd) {
    return 20.0f * log10(sqrtf(fd.x * fd.x + fd.y * fd.y));
}


__device__ void line_interp(int* y_max, int* y_max, const float2* d_signal, const int x, const int height, const float scale){
    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = (x>0)? db_abs(d_signal[x-1]) : abs_x_mid;
    float abs_x_right = (x<width-1)? db_abs(d_signal[x+1]) : abs_x_mid;
    int y_mid = int( scale * abs_x_mid + height/2);
    int left_y = int( 0.5 * scale * (abs_x_left + abs_x_mid) + height/2);
    int right_y = int( 0.5 * scale * (abs_x_right + abs_x_mid) + height/2);
    y_max = max(max(left_y, y_mid),right_y);
    y_min = min(min(left_y, y_mid),right_y);
}


__global__ void polchrome_kernel(const float2* f_domain, uchar4 *ptr, const short n_bins, const int n_spec) {
    //one block per bin
    //32 threads per block
    //we are heavly memory constricted

    assert(gridDim.x == 1024);
    assert(blockDim.x == 32);

    const int bin_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int num_threads = 32;
    const unsigned int height = 512;

    __shared__ short hist_column[num_threads*height];

    for(int y_ind=0; y_ind<height; y_ind++) hist_column[thread_idx* height + y_ind] = 0;

    int idx= bin_idx + thread_idx*n_bins;
    while (idx < n_bins*n_spec) {

        const float scale = 2.0f;
        int y_min, y_max;
        line_interp(y_min, y_max, d_signal, x, height, scale);

        hist_column[thread_idx*height + y_min] ++;
        hist_column[thread_idx*height + y_max] --

        idx += num_threads*n_bins;
    }

    __syncthreads();

    //collect cache fom threads / reduce
    int i = num_threads/2;
    while (i != 0) {
        if (thread_idx < i){
            for(int y_ind=0; y_ind<height; y_ind++) hist_column[thread_idx* height + y_ind] += hist_column[(thread_idx + i)*height+y_ind];
        }
        __syncthreads();
        i /= 2;
    }

    //integrate
    if (thread_idx == 0){
       for(int h=1; h<height; h++){
            hist[bin_idx  + h*n_bins] += hist[bin_idx  + (h-1)*n_bins];
    }

    //map
    if (thread_idx == 0){
       for(int h=0; h<height; h++) hist[bin_idx  + h * n_bins] = mapping(hist_column[h], n_spec);
    }

    //TODO delete me!! Make proper unit tests
    assert(mapping(0)==uchar4(0,0,0,0))
    assert(mapping(1)==uchar4(0,1,0,0))
    assert(mapping(n_spec)==uchar4(0,0,1,0))
}


void polchrome(float2* f_domain, unsigned short* hist, const int block_len, const int n_blocks){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const int blockSize = 32;
    const int numBlocks = n_bins;
    polchrome_kernel<<<numBlocks, blockSize>>>(f_domain, hist, block_len, n_blocks);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Timing end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    static int disp_count = 0;
    if((disp_count++)%100 == 0) printf("FFT postproc time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}