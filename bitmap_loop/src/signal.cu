#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"
#include "bitmap.h"

__global__ void generatePhasorSignal(float2* signal, int length, float omega, float phi, float noiseVariance, unsigned long long seed) {
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state);
    while (idx < length) {
        float noiseReal = curand_normal(&state) * noiseVariance;
        float noiseImag = curand_normal(&state) * noiseVariance;
        float angle = omega * idx + phi;
        signal[idx].x = cosf(angle) + noiseReal; // Real part
        signal[idx].y = sinf(angle) + noiseImag; // Imaginary part
        idx += blockDim.x  * gridDim.x;
    }
}

void generate_signal(float2* d_signal, const float phi, const int length, const int frame){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int blockSize = 256;
    float omega = 0.1f * 3.14159265359f; // 5 cycles over the signal
    int numBlocks = 256;
    float noiseVariance = 0.5f;
    generatePhasorSignal<<<numBlocks, blockSize>>>(d_signal, length, omega, phi, noiseVariance, (unsigned long long) frame);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Timing end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    static int disp_count = 0;
    if((disp_count++)%100 == 0) printf("Signal generation time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_fft(float2* t_domain, float2* f_domain, int length, int count) {
    static cufftHandle plan;
    static bool init = true;
    cufftResult result;

    // Create a 1D FFT plan for complex-to-complex (single precision)
    if (init){
        result = cufftPlan1d(&plan, length, CUFFT_C2C, count);
        assert(result == CUFFT_SUCCESS);
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
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    static int disp_count = 0;
    if((disp_count++)%100 == 0) printf("FFT calculation time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cufftDestroy(plan);
}

__global__ void fft_detector(const float2* f_domain, float* f_max, float* f_min, float* f_mean, const int block_len, const int n_blocks) {
    //one block per bin


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

void fft_postproc(float2* f_domain, uchar4* bitmap, const int block_len, const int n_blocks, int width, int height){
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
    const int numBlocks = n_bins;
    fft_detector<<<numBlocks, blockSize>>>(f_domain, f_max, f_min, f_mean, block_len, n_blocks);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    <<<numBlocks, blockSize>>>fill_bitmap_spec(bitmap, width, height, f_max, 1, false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    <<<numBlocks, blockSize>>>fill_bitmap_spec(bitmap, width, height, f_min, 1, false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    const int blockSize = 1024;
    const int numBlocks = n_bins;
    <<<numBlocks, blockSize>>>fill_bitmap_spec(bitmap, width, height, f_mean, 1, false);
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