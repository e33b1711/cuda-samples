#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#include "aux.h"
#include "fft.h"
#include "params.h"

__device__ uchar4 mapping(unsigned short hist_count, const int n_spec, uchar4 old)
{

    if (hist_count > n_spec)
    {
        printf("hist_count: %f\n", float(hist_count) / float(n_spec));
        // assert(false);
    }

    // 0 => 0 0 0 0
    if (hist_count == 0)
        return make_uchar4(old.x * 0.99, old.y * 0.99, old.z * 0.99, 0);

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

__device__ void line_interp(int &y_max, int &y_min, const float2 *d_signal, const int x, const ps params)
{
    float abs_x_mid = db_abs(d_signal[x]);
    float abs_x_left = abs_x_mid;
    float abs_x_right = abs_x_mid;
    if ((x % params.width) != 0)
        abs_x_left = db_abs(d_signal[x - 1]);
    if (((x + 1) % params.width) != 0)
        abs_x_right = db_abs(d_signal[x + 1]);
    int y_mid = int(params.scale * abs_x_mid + params.height / 2);
    int left_y = int(0.5 * params.scale * (abs_x_left + abs_x_mid) + params.height / 2);
    int right_y = int(0.5 * params.scale * (abs_x_right + abs_x_mid) + params.height / 2);
    y_max = max(max(left_y, y_mid), right_y);
    y_min = min(min(left_y, y_mid), right_y);
}

__global__ void polchrome_kernel(const float2 *f_domain, short *hist_unred, const ps params, const int width_unred)
{

    const int num_threads = blockDim.x;

    assert(num_threads <= params.block_len);

    const int height_max = 1024;
    assert(params.height <= height_max);
    short hist_column[height_max];

    for (int y_ind = 0; y_ind < params.height; y_ind++)
        hist_column[y_ind] = 0;

    const int thread_idx = threadIdx.x + blockIdx.x * num_threads;

    assert(width_unred > thread_idx);

    for (int t_idx = thread_idx; t_idx < params.block_len * params.n_blocks; t_idx += width_unred)
    {

        int y_min, y_max;
        line_interp(y_max, y_min, f_domain, t_idx, params);

        if (y_min < 0)
            y_min = 0;
        if (y_max < 0)
            continue;

        if (y_min < params.height)
            hist_column[y_min]++;

        if (y_max + 1 < params.height)
            hist_column[y_max + 1]--;
    }

    // integrate
    short accu = 0;
    for (int y_ind = 0; y_ind < params.height; y_ind++)
    {
        accu += hist_column[y_ind];
        hist_unred[y_ind + thread_idx * params.height] = accu;
    }
}

__global__ void polchrome_reduce(short *hist_unred, uchar4 *bitmap, const ps params, const int width_unred, const int reduce)
{

    assert(blockDim.x == params.height);
    assert(gridDim.x <= params.width);

    int y_ind = threadIdx.x;
    int x_ind = blockIdx.x;

    short hist = 0;
    for (int red = 0; red < reduce; red++)
    {
        hist += hist_unred[y_ind + x_ind * params.height + red * params.width * params.height];
    }
    bitmap[y_ind * params.width + x_ind] = mapping(hist, params.n_blocks, bitmap[y_ind * params.width + x_ind]);
}

void polchrome(const context ctx, const ps params)
{

    static short *hist_unred = nullptr;
    static uchar4 *internal_bitmap = nullptr;

    const int num_threads = min(512, params.block_len);
    const int num_blocks = 64;
    assert(params.block_len % num_threads == 0);

    const int overal_threads = num_threads * num_blocks;
    const int reduce = overal_threads / params.block_len;

    if (ctx.init)
    {
        CUDA_SAFE_CALL(cudaFree(hist_unred));
        CUDA_SAFE_CALL(cudaFree(internal_bitmap));
        CUDA_SAFE_CALL(cudaMalloc(&hist_unred, overal_threads * params.height * sizeof(short)));
        CUDA_SAFE_CALL(cudaMalloc(&internal_bitmap, params.width * params.height * sizeof(uchar4)));
        printf("init polychrome\n");
        printf("overal_threads: %d\n", overal_threads);
        printf("width: %d\n", params.width);
        printf("reduce: %d\n", reduce);
        printf("height: %d\n", params.height);
    }

    polchrome_kernel<<<num_blocks, num_threads, 0, ctx.stream>>>(ctx.f_domain, hist_unred, params, overal_threads);
    //CUDA_SAFE_CALL(cudaGetLastError());
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    polchrome_reduce<<<params.width, params.height, 0, ctx.stream>>>(hist_unred, internal_bitmap, params, overal_threads, reduce);
    //CUDA_SAFE_CALL(cudaGetLastError());
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.bitmap, internal_bitmap, params.width * params.height * sizeof(uchar4), cudaMemcpyDeviceToDevice, ctx.stream));
}