#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"

__global__ void generatePhasorSignal(float2 *signal, int length, float omega, float phi, float noiseVariance, unsigned long long seed, int spike_index)
{
    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state);
    while (idx < length)
    {
        float noiseReal = curand_normal(&state) * noiseVariance;
        float noiseImag = curand_normal(&state) * noiseVariance;
        float angle = omega * idx;
        signal[idx].x = cosf(angle) + noiseReal; // Real part
        signal[idx].y = sinf(angle) + noiseImag; // Imaginary part
        if (idx == (spike_index % length))
        {
            signal[idx].x = 20.0;
            signal[idx].y = 20.0;
        }
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void inject(float2 *signal, int block_len, int n_blocks, float2 value, int repetions, int index)
{
    int rep = 0;
    for(int ind=index; ind<n_blocks*block_len; ind += block_len){
        signal[ind] = value;
        if ((rep++) >= repetions) break;
    }
}

void generate_signal(cudaStream_t stream, float2 *d_signal, const float phi, const int length, const int frame)
{

    int blockSize = 256;
    float omega = 0.1f * 3.14159265359f; // 5 cycles over the signal
    int numBlocks = 256;
    float noiseVariance = 0.5f;
    generatePhasorSignal<<<numBlocks, blockSize, 0, stream>>>(d_signal, length, omega, phi, noiseVariance, (unsigned long long)frame, rand());
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void inject_spike(cudaStream_t stream, float2* f_domain, const int block_len, const int n_blocks, float2 value, int repetetions, int index){
    
    inject<<<1, 1, 0, stream>>>(f_domain, block_len, n_blocks, value, repetetions, index);
}