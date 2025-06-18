#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include "signal.h"

__device__ float generateRandomGaussian(float mean, float stddev, unsigned long long *seed) {
    curandState state;
    curand_init(*seed, 0, 0, &state);
    return mean + stddev * curand_normal(&state);
}

__global__ void generateComplexSignal(float2 *signal, int numSamples, float mean, float stddev, unsigned long long *seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        float real = generateRandomGaussian(mean, stddev, seed);
        float imag = generateRandomGaussian(mean, stddev, seed);
        signal[idx] = make_float2(real, imag);
    }
}

__global__ void addAWGN(float2 *signal, int numSamples, float noisePower, unsigned long long *seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        float noiseReal = generateRandomGaussian(0.0f, sqrtf(noisePower), seed);
        float noiseImag = generateRandomGaussian(0.0f, sqrtf(noisePower), seed);
        signal[idx].x += noiseReal;
        signal[idx].y += noiseImag;
    }
}

void generateComplexSignal(float2 *signal, int numSamples, float mean, float stddev) {
    unsigned long long seed = 1234; // Example seed
    int blockSize = 256;
    int numBlocks = (numSamples + blockSize - 1) / blockSize;
    generateComplexSignal<<<numBlocks, blockSize>>>(signal, numSamples, mean, stddev, &seed);
    cudaDeviceSynchronize();
}

void addAWGN(float2 *signal, int numSamples, float noisePower) {
    unsigned long long seed = 1234; // Example seed
    int blockSize = 256;
    int numBlocks = (numSamples + blockSize - 1) / blockSize;
    addAWGN<<<numBlocks, blockSize>>>(signal, numSamples, noisePower, &seed);
    cudaDeviceSynchronize();
}