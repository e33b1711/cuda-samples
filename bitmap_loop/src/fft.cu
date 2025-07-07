#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>

#include "aux.h"
#include "params.h"

void run_fft(cudaStream_t stream, float2 *t_domain, float2 *f_domain, const ps params)
{
    static cufftHandle plan;
    static bool init = true;
    cufftResult result;

    // Create a 1D FFT plan for complex-to-complex (single precision)
    if (init)
    {
        result = cufftPlan1d(&plan, params.block_len, CUFFT_C2C, params.n_blocks);
        assert(result == CUFFT_SUCCESS);
        cufftSetStream(plan, stream); // Associate the plan with the given stream
        init = false;
    }

    // Execute FFT (forward transform)
    result = cufftExecC2C(plan, (cufftComplex *)t_domain, (cufftComplex *)f_domain, CUFFT_FORWARD);
    assert(result == CUFFT_SUCCESS);
}

__global__ void fft_detector(const float2 *f_domain, float *f_max, float *f_min, float *f_mean, const ps params)
{

    const int num_blocks = gridDim.x;
    const int num_threads = blockDim.x;

    assert(num_threads <= params.block_len);
    //todo reduce if threads > bins


    const int thread_idx = threadIdx.x + blockIdx.x*num_threads;


    float max_v = -1e99f;
    float min_v = 1e99f;
    float mean_v = 0.0f;
 
    int t_idx = thread_idx;
    while (t_idx < params.block_len *  params.n_blocks)
    {
        float2 fd = f_domain[t_idx];
        float abs_v = sqrtf(fd.x * fd.x + fd.y * fd.y);
        max_v = max(max_v, abs_v);
        min_v = min(min_v, abs_v);
        mean_v += abs_v;

        t_idx += num_blocks * num_threads;
    }

    f_max[thread_idx] = max_v;
    f_min[thread_idx] = min_v;
    f_mean[thread_idx] = mean_v / params.n_blocks;
    
}


__global__ void fft_detector_reduce(float *f_max, float *f_min, float *f_mean, const ps params, const int n_threads)
{
    assert(gridDim.x == 1);
    assert(blockDim.x == params.block_len);

    float max_v = -1e99f;
    float min_v = 1e99f;
    float mean_v = 0.0f;


    for(int i = threadIdx.x; i< n_threads; i+=params.block_len){
        max_v = max(max_v, f_max[i]);
        min_v = min(min_v, f_min[i]);
        mean_v += f_mean[i];
    }

    f_max[threadIdx.x] = max_v;
    f_min[threadIdx.x] = min_v;
    f_mean[threadIdx.x] = mean_v;

  
}

__device__ float db_abs(float d_signal)
{
    return 20.0f * log10(d_signal);
}

__device__ void line_interp(int *y_min, int *y_max, const float *d_signal, const int x, const ps params)
{
    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = (x > 0) ? db_abs(d_signal[x - 1]) : abs_x_mid;
    float abs_x_right = (x < params.width - 1) ? db_abs(d_signal[x + 1]) : abs_x_mid;
    int y_mid = int(params.scale * abs_x_mid + params.height / 2);
    int left_y = int(0.5 * params.scale * (abs_x_left + abs_x_mid) + params.height / 2);
    int right_y = int(0.5 * params.scale * (abs_x_right + abs_x_mid) + params.height / 2);
    *y_max = max(max(left_y, y_mid), right_y);
    *y_min = min(min(left_y, y_mid), right_y);
}

__global__ void fill_bitmap_spec(uchar4 *ptr, const ps params, float *d_signal, int color, bool clear)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height)
        return;
    int idx = y * params.width + x;

    if (clear)
        ptr[idx] = make_uchar4(0, 0, 0, 0);

    int y_min, y_max;
    line_interp(&y_min, &y_max, d_signal, x, params);

    if (y <= y_max and y >= y_min)
    {
        if (color == 0)
            ptr[idx] = make_uchar4(255, 0, 0, 0);
        if (color == 1)
            ptr[idx] = make_uchar4(0, 255, 0, 0);
        if (color == 2)
            ptr[idx] = make_uchar4(0, 0, 255, 0);
        if (color == 3)
            ptr[idx] = make_uchar4(255, 255, 255, 0);
    }
}

void fft_postproc(cudaStream_t stream, float2 *f_domain, uchar4 *bitmap, const ps params)
{

    static float *f_max = nullptr;
    static float *f_min = nullptr;
    static float *f_mean = nullptr;
    static bool init = true;

    const int numThreads = 512;
    const int numBlocks = 64;
    assert(numThreads < params.block_len);

    if (init)
    {
        init = false;
        CUDA_SAFE_CALL(cudaMalloc(&f_max, numThreads * numBlocks * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&f_min, numThreads * numBlocks * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&f_mean, numThreads * numBlocks * sizeof(float)));
    }

    fft_detector<<<numBlocks, numThreads, 0, stream>>>(f_domain, f_max, f_min, f_mean, params);

    fft_detector_reduce<<<1, params.block_len, 0, stream>>>(f_max, f_min, f_mean, params, numThreads * numBlocks);

    dim3 block(16, 16);
    dim3 grid((params.width + block.x - 1) / block.x, (params.height + block.y - 1) / block.y);
    fill_bitmap_spec<<<grid, block, 0, stream>>>(bitmap, params, f_max, 3, false);

    fill_bitmap_spec<<<grid, block>>>(bitmap, params, f_min, 3, false);

    fill_bitmap_spec<<<grid, block>>>(bitmap, params, f_mean, 3, false);

}