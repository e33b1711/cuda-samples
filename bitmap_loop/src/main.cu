#include <cuda_runtime.h>
#include <unistd.h>

#include "gl_draw.h"
#include "aux.h"
#include "fft.h"
#include "signal.h"
#include "polychrome.h"


int main(int argc, char **argv) {

    const int BLOCK_LEN = 1024;
    const int N_BLOCKS = 1024*16;
    const int WIDTH = BLOCK_LEN;
    const int HEIGHT = 512;

    int frame = 0;

    draw_init(HEIGHT, WIDTH, argc, argv);

    cudaStream_t stream_ping, stream_pong;
    CUDA_SAFE_CALL( cudaStreamCreate( &stream_ping));
    CUDA_SAFE_CALL( cudaStreamCreate( &stream_pong));


    float2* t_domain_ping = nullptr;
    float2* t_domain_pong = nullptr;
    float2* t_domain_host = nullptr;
    float2* f_domain_ping = nullptr;
    float2* f_domain_pong = nullptr;
    uchar4* bitmap_ping = nullptr;
    uchar4* bitmap_pong = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain_ping, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&t_domain_pong, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    t_domain_host = (float2 *) malloc(BLOCK_LEN * N_BLOCKS * sizeof(float2));
    CUDA_SAFE_CALL(cudaMalloc(&f_domain_ping, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&f_domain_pong, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&bitmap_ping, WIDTH * HEIGHT * sizeof(uchar4)));
    CUDA_SAFE_CALL(cudaMalloc(&bitmap_pong, WIDTH * HEIGHT * sizeof(uchar4)));

    for(int offset=0; offset<2; offset++){
        generate_signal(stream_ping, t_domain_pong, 0.0f * float(frame), BLOCK_LEN*N_BLOCKS, frame);
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream_ping ) );
        size_t offset_s = offset * BLOCK_LEN * N_BLOCKS;
        CUDA_SAFE_CALL(cudaMemcpy(t_domain_host + offset_s, t_domain_pong, BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaMemcpyDeviceToHost));
    }

    while (true) {

        size_t offset_s = rand() % BLOCK_LEN * N_BLOCKS;
        CUDA_SAFE_CALL(cudaMemcpyAsync(t_domain_ping, t_domain_host + offset_s, BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaMemcpyDeviceToDevice, stream_ping));        
        run_fft(stream_pong, t_domain_pong, f_domain_pong, BLOCK_LEN, N_BLOCKS);
        //polchrome(stream_ping, f_domain_ping, bitmap_ping, BLOCK_LEN, N_BLOCKS, WIDTH);
        fft_postproc(stream_ping, f_domain_ping, bitmap_ping, BLOCK_LEN, N_BLOCKS, WIDTH, HEIGHT);
        //draw_loop(bitmap_pong, WIDTH, HEIGHT);
        time_info(BLOCK_LEN, N_BLOCKS);
        frame++;
        //usleep(1e6);

        CUDA_SAFE_CALL( cudaStreamSynchronize( stream_ping ) );
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream_pong ) );

        offset_s = rand() % BLOCK_LEN * N_BLOCKS;
        CUDA_SAFE_CALL(cudaMemcpyAsync(t_domain_pong, t_domain_host + offset_s, BLOCK_LEN * N_BLOCKS * sizeof(float2), cudaMemcpyDeviceToDevice, stream_pong));        
        run_fft(stream_ping, t_domain_ping, f_domain_ping, BLOCK_LEN, N_BLOCKS);
        //polchrome(stream_pong, f_domain_pong, bitmap, BLOCK_LEN, N_BLOCKS, WIDTH);
        fft_postproc(stream_pong, f_domain_pong, bitmap_pong, BLOCK_LEN, N_BLOCKS, WIDTH, HEIGHT);
        //draw_loop(bitmap_ping, WIDTH, HEIGHT);
        time_info(BLOCK_LEN, N_BLOCKS);
        frame++;

        CUDA_SAFE_CALL( cudaStreamSynchronize( stream_ping ) );
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream_pong ) );


    }

    draw_cleanup();
    return 0;
}