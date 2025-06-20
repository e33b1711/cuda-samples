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

void run_fft(float2* d_signal, int length) {
    cufftHandle plan;
    cufftResult result;

    // Create a 1D FFT plan for complex-to-complex (single precision)
    result = cufftPlan1d(&plan, length, CUFFT_C2C, 1);
    assert(result == CUFFT_SUCCESS);

    // Execute FFT in-place (forward transform)
    result = cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD);
    assert(result == CUFFT_SUCCESS);

    // Destroy the plan
    cufftDestroy(plan);
}