#include <cuda_runtime.h>

#include "gl_draw.h"
#include "aux.h"
#include "signal.h"
#include "polychrome.h"


int main(int argc, char **argv) {

    const int BLOCK_LEN = 1024;
    const int N_BLOCKS = 1024*16;
    const int WIDTH = SIGNAL_LENGTH;
    const int HEIGHT = 512;

    int frame = 0;

    draw_init(HEIGHT, WIDTH);

    float2* t_domain = nullptr;
    float2* f_domain = nullptr;
    uchar4* bitmap = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&f_domain, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&bitmap, WIDTH * HEIGHT * sizeof(uchar4)));

    while (true) {

        generate_signal(t_domain, 0.0f * float(frame), BLOCK_LEN*N_BLOCKS, frame);
        run_fft(t_domain, f_domain, BLOCK_LEN, N_BLOCKS);
        polchrome(f_domain, bitmap, BLOCK_LEN, N_BLOCKS);
        fft_postproc(f_domain, spec_bitmap, BLOCK_LEN, N_BLOCKS);
        draw_loop(bitmap, WIDTH, HEIGHT)
        time_info();
        frame++;
        //usleep(1e6);
    }

    draw_cleanup();
    return 0;
}