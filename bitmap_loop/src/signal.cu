#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h> // Add this include

#include "aux.h"

__global__ void addAWGN(float2* signal, float noiseVariance, int length, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float noiseReal = curand_normal(&state) * sqrt(noiseVariance);
        float noiseImag = curand_normal(&state) * sqrt(noiseVariance);
        signal[idx].x += noiseReal;
        signal[idx].y += noiseImag;
    }
}

__global__ void generatePhasorSignal(float2* signal, int length, float omega, float phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        float angle = omega * idx + phi;
        signal[idx].x = cosf(angle); // Real part
        signal[idx].y = sinf(angle); // Imaginary part
    }
}

void generate_signal(float2* d_signal, const float phi, const int length, const int frame){
    int blockSize = 256;
    float omega = 0.1f * 3.14159265359f; // 5 cycles over the signal
    int numBlocks = (length + blockSize - 1) / blockSize;
    generatePhasorSignal<<<numBlocks, blockSize>>>(d_signal, length, omega, phi);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    float noiseVariance = 0.5f;
    addAWGN<<<numBlocks, blockSize>>>(d_signal, noiseVariance, length, (unsigned long long) frame);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void run_fft(float2* t_domain, float2* f_domain, int length, int count) {
    cufftHandle plan;
    cufftResult result;

    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Create a 1D FFT plan for complex-to-complex (single precision)
    result = cufftPlan1d(&plan, length, CUFFT_C2C, count);
    assert(result == CUFFT_SUCCESS);

    // Execute FFT (forward transform)
    result = cufftExecC2C(plan, (cufftComplex*)t_domain, (cufftComplex*)f_domain, CUFFT_FORWARD);
    assert(result == CUFFT_SUCCESS);

    // Timing end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("FFT calculation time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cufftDestroy(plan);
}

__global__ void fft_detector(float2* f_domain, float* f_max, float* f_min, float* f_mean, int length, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    float max_v = -1e99f;
    float min_v = 1e99f;
    float mean_v = 0.0f;
    int idx_up = idx;
    while (idx_up < length*count) {
        float abs_v = sqrt( pow(f_domain[idx_up].x, 2) + pow(f_domain[idx_up].y, 2) ); 
        max_v = max(max_v, abs_v);
        min_v = min(min_v, abs_v);
        mean_v += abs_v; 
        idx_up += length;
    }
    f_max[idx] = max_v;
    f_min[idx] = min_v;
    f_mean[idx] = mean_v / count;
}

void fft_postproc(float2* f_domain, float* f_max, float* f_min, float* f_mean, const int length, const int count){
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    fft_detector<<<numBlocks, blockSize>>>(f_domain, f_max, f_min, f_mean, length, count);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}