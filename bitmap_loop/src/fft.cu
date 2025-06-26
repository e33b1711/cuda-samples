#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"


void run_fft(cudaStream_t stream, float2* t_domain, float2* f_domain, int length, int count) {
    static cufftHandle plan;
    static bool init = true;
    cufftResult result;

    // Create a 1D FFT plan for complex-to-complex (single precision)
    if (init){
        result = cufftPlan1d(&plan, length, CUFFT_C2C, count);
        assert(result == CUFFT_SUCCESS);
        cufftSetStream(plan, stream); // Associate the plan with the given stream
    }

     // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Execute FFT (forward transform)
    result = cufftExecC2C(plan, (cufftComplex*)t_domain, (cufftComplex*)f_domain, CUFFT_FORWARD);
    assert(result == CUFFT_SUCCESS);

    // Timing end
    cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    static int disp_count = 0;
    if((disp_count++)%100 == 0) printf("FFT calculation time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //cufftDestroy(plan);
}


__global__ void fft_detector(const float2* f_domain, float* f_max, float* f_min, float* f_mean, const int block_len, const int n_blocks) {

    const int bin_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_threads_max = 1024;
    assert(num_threads <= num_threads_max);

    __shared__ float max_cache[num_threads_max];
    __shared__ float min_cache[num_threads_max];
    __shared__ float mean_cache[num_threads_max];

    float max_v = -1e99f;
    float min_v = 1e99f;
    float mean_v = 0.0f;
    int idx= bin_idx + thread_idx*block_len;
    while (idx < block_len*n_blocks) {
        float2 fd = f_domain[idx];
        float abs_v = sqrtf(fd.x * fd.x + fd.y * fd.y);
        max_v = max(max_v, abs_v);
        min_v = min(min_v, abs_v);
        mean_v += abs_v;

        idx += block_len*num_threads;
    }
    max_cache[thread_idx] = max_v;
    min_cache[thread_idx] = min_v;
    mean_cache[thread_idx] = mean_v;

    __syncthreads();

    int i = num_threads/2;
    while (i != 0) {
        if (thread_idx < i){
            max_cache[thread_idx] = max(max_cache[thread_idx], max_cache[thread_idx + i]);
            min_cache[thread_idx] = min(min_cache[thread_idx], min_cache[thread_idx + i]);
            mean_cache[thread_idx] += mean_cache[thread_idx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (thread_idx == 0){
       f_max[bin_idx] = max_cache[0];
       f_min[bin_idx] = min_cache[0];
       f_mean[bin_idx] = mean_cache[0] / n_blocks;
    }
}


__device__ float db_abs(float d_signal) {
    return 20.0f * log10(d_signal);
}


__device__ void line_interp(int* y_min, int* y_max, const float* d_signal, const int x, const int height, const int width, const float scale){
    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = (x>0)? db_abs(d_signal[x-1]) : abs_x_mid;
    float abs_x_right = (x<width-1)? db_abs(d_signal[x+1]) : abs_x_mid;
    int y_mid = int( scale * abs_x_mid + height/2);
    int left_y = int( 0.5 * scale * (abs_x_left + abs_x_mid) + height/2);
    int right_y = int( 0.5 * scale * (abs_x_right + abs_x_mid) + height/2);
    *y_max = max(max(left_y, y_mid),right_y);
    *y_min = min(min(left_y, y_mid),right_y);
}


__global__ void fill_bitmap_spec(uchar4* ptr, int width, int height, float* d_signal, int color, bool clear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    if (clear) ptr[idx] = make_uchar4(0,0,0,0);

    const float scale = 2.0f;
    int y_min, y_max;
    line_interp(&y_min, &y_max, d_signal, x, height, width, scale);

    if (y <= y_max and y >= y_min){
        if (color==0) ptr[idx] = make_uchar4(255,0,0,0);
        if (color==1) ptr[idx] = make_uchar4(0,255,0,0);
        if (color==2) ptr[idx] = make_uchar4(0,0,255,0);
        if (color==3) ptr[idx] = make_uchar4(255,255,255,0);
    }
}
    

void fft_postproc(cudaStream_t stream, float2* f_domain, uchar4* bitmap, const int block_len, const int n_blocks, int width, int height){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    static float* f_max = nullptr;
    static float* f_min = nullptr;
    static float* f_mean = nullptr;
    static bool init = true;

    if(init){
        init = false;
        CUDA_SAFE_CALL(cudaMalloc(&f_max, block_len * n_blocks * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&f_min, block_len * n_blocks * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&f_mean, block_len * n_blocks * sizeof(float)));
    }

    const int blockSize = 1024;
    const int numBlocks = block_len;
    fft_detector<<<numBlocks, blockSize, 0, stream>>>(f_domain, f_max, f_min, f_mean, block_len, n_blocks);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fill_bitmap_spec<<<grid, block, 0, stream>>>(bitmap, width, height, f_max, 3, false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    fill_bitmap_spec<<<grid, block>>>(bitmap, width, height, f_min, 3, false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    fill_bitmap_spec<<<grid, block>>>(bitmap, width, height, f_mean, 3, false);
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