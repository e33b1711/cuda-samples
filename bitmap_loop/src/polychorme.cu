#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#include "aux.h"

//TODO log mapping + color sheme

//TODO line interp




__global__ void polchrome_kernel(const float2* f_domain, unsigned short* hist, const short n_bins, const int n_spec) {
    //one block per bin
    //32 threads per block
    //we are heavly memory constricted

    assert(gridDim.x == 1024);
    assert(blockDim.x == 32);


    const int bin_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int num_threads = 32;
    const unsigned int height = 512;

    __shared__ unsigned short hist_column[num_threads*height];

    for(int y_ind=0; y_ind<height; y_ind++) hist_column[thread_idx* height + y_ind] = 0;

    int idx= bin_idx + thread_idx*n_bins;
    while (idx < n_bins*n_spec) {

        float2 fd = f_domain[idx];
        float abs_db = 20.0f * log10(sqrtf(fd.x * fd.x + fd.y * fd.y));
        const float scale = 2.0f;
        int unsigned y_ind = int( scale * abs_db + height/2) % height;

        hist_column[thread_idx*height + y_ind] ++;

        idx += num_threads*n_bins;
    }

    __syncthreads();

    int i = num_threads/2;
    while (i != 0) {
        if (thread_idx < i){
            for(int y_ind=0; y_ind<height; y_ind++) hist_column[thread_idx* height + y_ind] += hist_column[(thread_idx + i)*height+y_ind];
        }
        __syncthreads();
        i /= 2;
    }

    if (thread_idx == 0){
       for(int h=0; h<height; h++) hist[bin_idx  + h * n_bins] = 20*(0.0 -log(float(hist_column[h]) / float(n_spec)));  
    }
}

void polchrome(float2* f_domain, unsigned short* hist, const int n_bins, const int n_spec){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const int blockSize = 32;
    const int numBlocks = n_bins;
    fft_detector<<<numBlocks, blockSize>>>(f_domain, hist, n_bins, n_spec);
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


__global__ void fill_bitmap_spec(uchar4 *ptr, int width, int height, int frame, unsigned short* hist, int color, bool clear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

        ptr[idx].x = hist[idx]%256;
        ptr[idx].z = 0;
        ptr[idx].y = 0;
        ptr[idx].w = 255;
    
}