#include <cuda_runtime.h>
#include <unistd.h>

#include "draw.h"
#include "aux.h"
#include "fft.h"
#include "signal.h"
#include "polychrome.h"
#include "params.h"
#include "udp_source.h"


void init(context &ctx, const ps params)
{
    CUDA_SAFE_CALL(cudaStreamCreate(&ctx.stream));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&ctx.bitmap_host, params.height * params.width * sizeof(uchar4), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.bitmap, params.height * params.width * sizeof(uchar4)));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.t_domain, params.block_len * params.n_blocks * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&ctx.f_domain, params.block_len * params.n_blocks * sizeof(float2)));
    CUDA_SAFE_CALL(cudaEventCreate(&ctx.start));
    CUDA_SAFE_CALL(cudaEventCreate(&ctx.stop));
    ctx.init = true;
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

    ping.init = false;
    pong.init = false;
}

float2 *init_host_signal(ps params)
{
    float2 *t_domain_host = nullptr;
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&t_domain_host, 2 * params.block_len * params.n_blocks * sizeof(float2), cudaHostAllocMapped));

    float2 *t_domain = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain, 2 * params.block_len * params.n_blocks * sizeof(float2)));
    generate_signal(t_domain, 2 * params.block_len * params.n_blocks);
    CUDA_SAFE_CALL(cudaMemcpy(t_domain_host, t_domain, 2 * params.block_len * params.n_blocks * sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(t_domain));
    return t_domain_host;
}

void handle_input(context ctx, float2 *t_domain_host, ps params)
{
    cudaEventRecord(ctx.start, ctx.stream);
    size_t offset_s = rand() % (params.block_len * params.n_blocks);
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.t_domain, t_domain_host, params.block_len * params.n_blocks * sizeof(float2), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.bitmap_host, ctx.bitmap, params.height * params.width * sizeof(uchar4), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_SAFE_CALL(cudaGetLastError());
    cudaEventRecord(ctx.stop, ctx.stream);
}

void handle_dsp(context ctx, ps params)
{
    cudaEventRecord(ctx.start, ctx.stream);
    run_fft(ctx, params);
    float2 value;
    value.x = float(rand() % 1000) / 0.01;
    value.y = float(rand() % 1000) / 0.01;
    if (rand() % 1000 < 0)
    {
        size_t offset_s = rand() % params.block_len;
        inject_spike(ctx.stream, ctx.f_domain, params.block_len, params.n_blocks, value, 1, offset_s);
    }
    polchrome(ctx, params);
    fft_postproc(ctx, params);
    cudaEventRecord(ctx.stop, ctx.stream);
}

void sync(context &ctx)
{
    cudaEventSynchronize(ctx.stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ctx.start, ctx.stop);
    CUDA_SAFE_CALL(cudaStreamSynchronize(ctx.stream));
}

void destroy(context &ctx)
{
    CUDA_SAFE_CALL(cudaEventDestroy(ctx.start));
    CUDA_SAFE_CALL(cudaEventDestroy(ctx.stop));
    CUDA_SAFE_CALL(cudaStreamDestroy(ctx.stream));
    CUDA_SAFE_CALL(cudaFree(ctx.t_domain));
    CUDA_SAFE_CALL(cudaFree(ctx.f_domain));
    CUDA_SAFE_CALL(cudaFree(ctx.bitmap));
    CUDA_SAFE_CALL(cudaFreeHost(ctx.bitmap_host));
}


float one_pass(ps params)
{

    int frame = 0;
    float throughput;
    context ping, pong;
    float2 *t_domain_host = nullptr;

    UdpSource udp(params.block_len * params.n_blocks);
    udp.init();

    init(ping, params);
    init(pong, params);

    while (frame < params.num_loops)
    {
        t_domain_host = udp.process_next_buffer();
        handle_input(ping, t_domain_host, params);
        handle_dsp(pong, params);
        throughput = time_info(params.block_len, params.n_blocks);
        drawImage(pong.bitmap_host, params);
        sync(ping);
        sync(pong);
        CUDA_SAFE_CALL(cudaGetLastError());
        switch_context(ping, pong);
        frame++;
    }

    destroy(ping);
    destroy(pong);

    return throughput;
}

int main(int argc, char **argv)
{
    

    float throughput[10];

    ps params;
    params.n_blocks = 256;
    params.num_loops = 500;
    throughput[0] = one_pass(params);

    params.block_len = 2048;
    params.width = 2048;
    throughput[1] = one_pass(params);

    params.block_len = 256;
    params.width = 256;
    params.height = 1024;
    throughput[2] = one_pass(params);

    for(int i=0; i<3; i++){
        printf("throughput: %f\n", throughput[i]);
    }


    return 0;
}