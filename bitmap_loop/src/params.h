#pragma once

    struct ps{
        int block_len = 1024;
        int n_blocks = 1024 * 16;
        int width = 1024;
        int height = 512;
        float scale = 2.0;
    };

struct context
{
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    //
    float2 *t_domain = nullptr;
    uchar4 *bitmap = nullptr;
    uchar4 *bitmap_host = nullptr;
    float2 *f_domain = nullptr;
    //
    bool init = true;
};