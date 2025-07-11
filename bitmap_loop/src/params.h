#pragma once

    struct ps{
        int block_len = 1024;
        int overlap_len = 512;
        int n_t_blocks = 1024 * 16;
        int n_f_blocks = 2 * 1024 * 16 -1;
        int width = 1024;
        int height = 512;
        float scale = 2.0;
        int num_loops = 1e3;
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

//todo depended params