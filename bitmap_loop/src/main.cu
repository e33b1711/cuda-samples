#include <cuda_runtime.h>
#include <unistd.h>

#include "gl_draw.h"
#include "aux.h"
#include "fft.h"
#include "signal.h"
#include "polychrome.h"

int main(int argc, char **argv)
{

    const int BLOCK_LEN = 1024;
    const int N_BLOCKS = 1024 * 16;
    const int WIDTH = BLOCK_LEN;
    const int HEIGHT = 512;

    int frame = 0;

    draw_init(HEIGHT, WIDTH, argc, argv);

    cudaStream_t stream_in, stream_dsp;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream_in));
    CUDA_SAFE_CALL(cudaStreamCreate(&stream_dsp));

    float2 *t_domain_ping = nullptr;
    float2 *t_domain_pong = nullptr;

    float2 *t_domain_host = nullptr;

    float2 *f_domain = nullptr;
    uchar4 *bitmap = nullptr;

    CUDA_SAFE_CALL(cudaMalloc(&t_domain_ping, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&t_domain_pong, BLOCK_LEN * N_BLOCKS * sizeof(float2)));

    CUDA_SAFE_CALL(cudaHostAlloc((void **)&t_domain_host, 2 * BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaHostAllocDefault));

    CUDA_SAFE_CALL(cudaMalloc(&f_domain, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&bitmap, WIDTH * HEIGHT * sizeof(uchar4)));

    for (int offset = 0; offset < 2; offset++)
    {
        generate_signal(stream_in, t_domain_ping, 0.0f * float(frame), BLOCK_LEN * N_BLOCKS, frame);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream_in));
        size_t offset_s = offset * BLOCK_LEN * N_BLOCKS;
        CUDA_SAFE_CALL(cudaMemcpy(t_domain_host + offset_s, t_domain_ping, BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaMemcpyDeviceToHost));
    }

    // Timing start
    cudaEvent_t dsp_start, dsp_stop;
    cudaEvent_t in_start, in_stop;
    cudaEventCreate(&dsp_start);
    cudaEventCreate(&dsp_stop);
    cudaEventCreate(&in_start);
    cudaEventCreate(&in_stop);

    while (true)
    {

        size_t offset_s = rand() % BLOCK_LEN * N_BLOCKS;
        float2 *t_domain_in = nullptr;
        float2 *t_domain_dsp = nullptr;
        if (frame % 2 == 0)
        {
            t_domain_in = t_domain_pong;
            t_domain_dsp = t_domain_ping;
        }
        else
        {
            t_domain_in = t_domain_ping;
            t_domain_dsp = t_domain_ping;
        }

        cudaEventRecord(in_start, stream_in);
        CUDA_SAFE_CALL(cudaMemcpyAsync(t_domain_in, t_domain_host + frame, BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaMemcpyHostToDevice, stream_in));
        cudaEventRecord(in_stop, stream_in);

        cudaEventRecord(dsp_start, stream_dsp);
        run_fft(stream_dsp, t_domain_dsp, f_domain, BLOCK_LEN, N_BLOCKS);
        float2 value; value.x = 1272.9; value.y = 2827.0;
        if (rand()%1000 < 20)
        inject_spike(stream_dsp, f_domain, BLOCK_LEN, N_BLOCKS, value,  34, rand()%BLOCK_LEN);
        polchrome(stream_dsp, f_domain, bitmap, BLOCK_LEN, N_BLOCKS, WIDTH, HEIGHT);
        fft_postproc(stream_dsp, f_domain, bitmap, BLOCK_LEN, N_BLOCKS, WIDTH, HEIGHT);
        cudaEventRecord(dsp_stop, stream_dsp);

        draw_loop(bitmap, WIDTH, HEIGHT);
        time_info(BLOCK_LEN, N_BLOCKS);
        frame++;
        //usleep(1e6);

        // Timing end
        cudaEventSynchronize(dsp_stop);
        cudaEventSynchronize(in_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, dsp_start, dsp_stop);
        static int disp_count = 0;
        if ((disp_count) % 100 == 0)
            printf("DSP time: %.3f ms\n", ms);
        cudaEventElapsedTime(&ms, in_start, in_stop);
        if ((disp_count) % 100 == 0)
            printf("IN time: %.3f ms\n", ms);
        cudaEventElapsedTime(&ms, in_start, dsp_stop);
        if ((disp_count++) % 100 == 0)
            printf("IN2DSP time: %.3f ms\n", ms);

        CUDA_SAFE_CALL(cudaStreamSynchronize(stream_dsp));
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream_in));
    }

    // Cleanup
    cudaEventDestroy(dsp_start);
    cudaEventDestroy(dsp_stop);
    cudaEventDestroy(in_start);
    cudaEventDestroy(in_stop);

    draw_cleanup();
    return 0;
}