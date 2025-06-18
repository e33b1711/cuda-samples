#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "draw.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

const int SIGNAL_LENGTH = 1024*1024;

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

int main() {
    float2* d_signal = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, SIGNAL_LENGTH * sizeof(float2)));

    int blockSize = 256;
    int numBlocks = (SIGNAL_LENGTH + blockSize - 1) / blockSize;

    float omega = 2.0f * 3.14159265359f * 5.0f / SIGNAL_LENGTH; // 5 cycles over the signal
    float phi = 0.0f;
    generatePhasorSignal<<<numBlocks, blockSize>>>(d_signal, SIGNAL_LENGTH, omega, phi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float noiseVariance = 0.01f;
    addAWGN<<<numBlocks, blockSize>>>(d_signal, noiseVariance, SIGNAL_LENGTH, 1234ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    displaySignalWithCudaInterop(d_signal, SIGNAL_LENGTH);

    CUDA_CHECK(cudaFree(d_signal));
    return 0;
}