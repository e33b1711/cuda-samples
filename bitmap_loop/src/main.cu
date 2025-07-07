#include <cuda_runtime.h>
#include <unistd.h>

#include "draw.h"
#include "aux.h"
#include "fft.h"
#include "signal.h"
#include "polychrome.h"
#include "params.h"

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
};

void init(context &ctx, const ps params)
{
    CUDA_SAFE_CALL(cudaStreamCreate(&ctx.stream));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&ctx.bitmap_host, params.height * params.width * sizeof(uchar4), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.bitmap, params.height * params.width * sizeof(uchar4)));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.t_domain, params.block_len * params.n_blocks * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.f_domain, params.block_len * params.n_blocks * sizeof(float2)));
    CUDA_SAFE_CALL(cudaEventCreate(&ctx.start));
    CUDA_SAFE_CALL(cudaEventCreate(&ctx.stop));
}

void switch_context(context &ping, context &pong)
{
    context temp = ping;

    ping.t_domain = pong.t_domain;
    ping.bitmap = pong.bitmap;
    ping.bitmap_host = pong.bitmap_host;
    ping.f_domain = pong.f_domain;

    pong.t_domain = temp.t_domain;
    pong.bitmap = temp.bitmap;
    pong.bitmap_host = temp.bitmap_host;
    pong.f_domain = temp.f_domain;
}

float2 *init_host_signal(ps params)
{
    float2 *t_domain_host = nullptr;
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&t_domain_host, 2 * params.block_len * params.n_blocks * sizeof(float2), cudaHostAllocMapped));

    float2 *t_domain = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain, 2 * params.block_len * params.n_blocks * sizeof(float2)));
    generate_signal(t_domain, 2 * params.block_len * params.n_blocks);
    CUDA_SAFE_CALL(cudaMemcpy(t_domain_host, t_domain, 2 * params.block_len * params.n_blocks * sizeof(float2), cudaMemcpyDeviceToHost));
    return t_domain_host;
}

void handle_input(context ctx, float2 *t_domain_host, ps params)
{
    cudaEventRecord(ctx.start, ctx.stream);
    size_t offset_s = rand() % (params.block_len * params.n_blocks);
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.t_domain, t_domain_host + offset_s, params.block_len * params.n_blocks * sizeof(float2), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.bitmap_host, ctx.bitmap, params.height * params.width * sizeof(uchar4), cudaMemcpyDeviceToHost, ctx.stream));
    cudaEventRecord(ctx.stop, ctx.stream);
}

void handle_dsp(context ctx, ps params)
{
    cudaEventRecord(ctx.start, ctx.stream);
    run_fft(ctx.stream, ctx.t_domain, ctx.f_domain, params);
    float2 value;
    value.x = float(rand() % 1000) / 0.01;
    value.y = float(rand() % 1000) / 0.01;
    if (rand() % 1000 < 20)
    {
        size_t offset_s = rand() % params.block_len;
        inject_spike(ctx.stream, ctx.f_domain, params.block_len, params.n_blocks, value, 34, offset_s);
    }
    polchrome(ctx.stream, ctx.f_domain, ctx.bitmap, params);
    fft_postproc(ctx.stream, ctx.f_domain, ctx.bitmap, params);
    cudaEventRecord(ctx.stop, ctx.stream);
}

void sync(context &ping, context &pong)
{ 
    cudaEventSynchronize(ping.stop);
    cudaEventSynchronize(pong.stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, pong.start, pong.stop);
    static int disp_count = 0;
    if ((disp_count) % 100 == 0)
        printf("DSP time: %.3f ms\n", ms);
    cudaEventElapsedTime(&ms, ping.start, ping.stop);
    if ((disp_count) % 100 == 0)
        printf("IN time: %.3f ms\n", ms);
    cudaEventElapsedTime(&ms, ping.start, ping.stop);
    if ((disp_count++) % 100 == 0)
        printf("IN2DSP time: %.3f ms\n", ms);

    CUDA_SAFE_CALL(cudaStreamSynchronize(ping.stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(pong.stream));
}

int main(int argc, char **argv)
{

    ps params;
    params.block_len = 1024;
    params.n_blocks = 16 * 1024;
    params.width = 1024;
    params.height = 512;
    params.scale = 2.0f;

    int frame = 0;

    context ping, pong;

    init(ping, params);
    init(pong, params);

    float2 *t_domain_host = init_host_signal(params);

    while (frame<500)
    {

        switch_context(ping, pong);
        handle_input(ping, t_domain_host, params);
        handle_dsp(pong, params);
        time_info(params.block_len, params.n_blocks);
        frame++;
        drawImage(pong.bitmap_host, params);
        sync(ping, pong);
    }

    // Cleanup
    cudaEventDestroy(ping.start);
    cudaEventDestroy(ping.stop);
    cudaEventDestroy(pong.start);
    cudaEventDestroy(pong.stop);

    // draw_cleanup();
    return 0;
}