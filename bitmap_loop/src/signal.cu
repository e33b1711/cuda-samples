#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h> // Add this include

#include "aux.h"

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

__global__ void fft_detector(const float2* f_domain, float* f_max, float* f_min, float* f_mean, const int n_bins, const int n_spec) {
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
    int idx= bin_idx + thread_idx*n_bins;
    while (idx < n_bins*n_spec) {
        float2 fd = f_domain[idx];
        float abs_v = sqrtf(fd.x * fd.x + fd.y * fd.y);
        max_v = max(max_v, abs_v);
        min_v = min(min_v, abs_v);
        mean_v += abs_v; 

        idx += n_bins*num_threads;
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
       f_mean[bin_idx] = mean_cache[0] / n_spec;
    }
}

void fft_postproc(float2* f_domain, float* f_max, float* f_min, float* f_mean, const int n_bins, const int n_spec){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const int blockSize = 1024;
    const int numBlocks = n_bins;
    fft_detector<<<numBlocks, blockSize>>>(f_domain, f_max, f_min, f_mean, n_bins, n_spec);
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