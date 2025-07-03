#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#include "aux.h"
#include "fft.h"

__device__ uchar4 mapping(unsigned short hist_count, const int n_spec)
{

    if (hist_count > n_spec)
    {
        printf("hist_count: %f\n", float(hist_count) / float(n_spec));
        assert(false);
    }

    // 0 => 0 0 0 0
    if (hist_count == 0)
        return make_uchar4(0, 0, 0, 0);

    // 1 => 0.0 / 0
    // n_spec => 1.0 / 510
    float color_index = log2f(hist_count) / log2f(n_spec);
    int c_index = int(510.0 * color_index);

    // index: 0-255
    //        0-255
    //        255-0
    //        0
    if (c_index < 256)
    {
        return make_uchar4(0, 255 - c_index, c_index, 0);
        // return make_uchar4(50, 50, 50, 0);
    }
    // index: 256-510
    //        254-0
    //        0
    //        1-255
    if (c_index < 511)
    {
        return make_uchar4(c_index - 255, 0, 510 - c_index, 0);
        // return make_uchar4(50, 50, 50, 0);
    }

    return make_uchar4(100, 100, 100, 0);
}

__device__ float db_abs(float2 fd)
{
    return 20.0f * log10(sqrtf(fd.x * fd.x + fd.y * fd.y));
}

__device__ void line_interp(int &y_max, int &y_min, const float2 *d_signal, const int x, const int height, const int width, const float scale)
{
    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = (x > 0) ? db_abs(d_signal[x - 1]) : abs_x_mid;
    float abs_x_right = (x < width - 1) ? db_abs(d_signal[x + 1]) : abs_x_mid;
    int y_mid = int(scale * abs_x_mid + height / 2);
    int left_y = int(0.5 * scale * (abs_x_left + abs_x_mid) + height / 2);
    int right_y = int(0.5 * scale * (abs_x_right + abs_x_mid) + height / 2);
    y_max = max(max(left_y, y_mid), right_y);
    y_min = min(min(left_y, y_mid), right_y);
}

__global__ void polchrome_kernel(const float2 *f_domain, short *hist_unred, const short block_len, const int n_blocks, const int width, const int height)
{

    const int num_blocks = gridDim.x;
    const int num_threads = blockDim.x;

    assert(num_threads <= block_len);

    const int height_max = 512;
    assert(height <= height_max);
    short hist_column[height_max];

    for (int y_ind = 0; y_ind < height; y_ind++)
        hist_column[y_ind] = 0;

    const int thread_idx = threadIdx.x + blockIdx.x * num_threads;

    int t_idx = thread_idx;

    while (t_idx < block_len * n_blocks)
    {

        const float scale = 2.0f;
        int y_min, y_max;
        line_interp(y_max, y_min, f_domain, t_idx, height, width, scale);

        if (y_min < 0)
            y_min = 0;
        if (y_max < 0)
        {
            y_min = height;
            y_max = height;
        };
        if (y_min < height)
            hist_column[y_min]++;
        if (y_max + 1 < height)
            hist_column[y_max + 1]--;

        t_idx += num_blocks * num_threads;
    }

    for (int y_ind = 0; y_ind < height; y_ind++)
        hist_unred[thread_idx*height + y_ind] = hist_column[y_ind];
}


__global__ void polchrome_reduce(short* hist_unred, uchar4* bitmap, const int width_unred, const int width, const int height, const int n_blocks){

    const int num_blocks = gridDim.x;
    const int num_threads = blockDim.x;

    assert(num_threads == width);

    const int height_max = 512;
    assert(height <= height_max);
    short hist_column[height_max];

    for (int y_ind = 0; y_ind < height; y_ind++)
        hist_column[y_ind] = 0;

    const int thread_idx = threadIdx.x + blockIdx.x * num_threads;

    //reduce
    for (int x_index =  thread_idx; x_index < width_unred; x_index+= width){
        for (int y_ind = 0; y_ind < height; y_ind++)
            hist_column[y_ind] += hist_unred[x_index*height + y_ind];
    }

    // integrate 
    for (int h = 1; h < height; h++)
    {
        hist_column[h] += hist_column[h - 1];
    }

     // map
    for (int h = 0; h < height; h++)
        bitmap[thread_idx + h * width] = mapping(hist_column[h], n_blocks);
}


void polchrome(cudaStream_t stream, float2 *f_domain, uchar4 *bitmap, const int block_len, const int n_blocks, const int width, const int height)
{




    static short *hist_unred = nullptr;
    static bool init = true;

    const int numThreads = 512;
    const int numBlocks = 64;
    assert(numThreads < block_len);

    if (init)
    {
        init = false;
        CUDA_SAFE_CALL(cudaMalloc(&hist_unred, numThreads * numBlocks * height * sizeof(short)));
    }



    polchrome_kernel<<<numBlocks, numThreads, 0, stream>>>(f_domain, hist_unred, block_len, n_blocks, width, height);
    polchrome_reduce<<<1, width, 0, stream>>>(hist_unred, bitmap, numThreads * numBlocks, width, height, n_blocks);
}