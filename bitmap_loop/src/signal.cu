#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"


__global__ void generatePhasorSignal(float2* signal, int length, float omega, float phi, float noiseVariance, unsigned long long seed, int spike_index) {
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state);
    while (idx < length) {
        float noiseReal = curand_normal(&state) * noiseVariance;
        float noiseImag = curand_normal(&state) * noiseVariance;
        float angle = omega * idx + phi;
        signal[idx].x = cosf(angle) + noiseReal; // Real part
        signal[idx].y = sinf(angle) + noiseImag; // Imaginary part
        if (idx==(spike_index % length)){
            signal[idx].x = 20.0;
            signal[idx].y = 20.0;
        }
        idx += blockDim.x  * gridDim.x;
    }
}


void generate_signal(cudaStream_t stream, float2* d_signal, const float phi, const int length, const int frame){
    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int blockSize = 256;
    float omega = 0.1f * 3.14159265359f; // 5 cycles over the signal
    int numBlocks = 256;
    float noiseVariance = 0.5f;
    generatePhasorSignal<<<numBlocks, blockSize, 0, stream>>>(d_signal, length, omega, phi, noiseVariance, (unsigned long long) frame, rand());


    // Timing end
    cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    static int disp_count = 0;
    if((disp_count++)%100 == 0) printf("Signal generation time: %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
