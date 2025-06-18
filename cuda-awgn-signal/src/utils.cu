#include <iostream>
#include <cuda_runtime.h>

__global__ void drawSignalKernel(const float2* signal, int numSamples) {
    // This kernel would contain the logic to visualize the signal.
    // For demonstration purposes, we will just print the signal values.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        printf("Sample %d: (%f, %f)\n", idx, signal[idx].x, signal[idx].y);
    }
}

void drawSignal(const float2* signal, int numSamples) {
    float2* d_signal;
    cudaMalloc(&d_signal, numSamples * sizeof(float2));
    cudaMemcpy(d_signal, signal, numSamples * sizeof(float2), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numSamples + blockSize - 1) / blockSize;
    drawSignalKernel<<<numBlocks, blockSize>>>(d_signal, numSamples);
    
    cudaDeviceSynchronize();
    cudaFree(d_signal);
}