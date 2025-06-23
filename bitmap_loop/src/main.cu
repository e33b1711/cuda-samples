#include "gl_draw.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono> // Add at the top
#include <unistd.h>

#include "aux.h"
#include "signal.h"
#include "polychrome.h"


const int BLOCK_LEN = 1024;
const int N_BLOCKS = 1024*16;
const int WIDTH = SIGNAL_LENGTH;
const int HEIGHT = 512;

cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;
int frame = 0; // Global frame counter


void launch_cuda_kernel(uchar4* bitmap) {
    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));

    CUDA_SAFE_CALL(cudaDeviceSynchronize(cudaMemcopy(dptr, bitmap, WIDTH * HEIGHT * sizeof(uchar4))));

    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, WIDTH * HEIGHT * sizeof(uchar4)));
}


void cleanup() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
}


void dummy_display() {}


int main(int argc, char **argv) {
    initGLUT(&argc, argv, cleanup, HEIGHT, WIDTH);
    glutDisplayFunc(dummy_display); // Register dummy display callback
    initPixelBuffer(&pbo, &tex, &cuda_pbo_resource, HEIGHT, WIDTH);

    float2* t_domain = nullptr;
    float2* f_domain = nullptr;
    uchar4* bitmap = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&f_domain, BLOCK_LEN * N_BLOCKS * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&bitmap, WIDTH * HEIGHT * sizeof(uchar4)));

    using clock = std::chrono::high_resolution_clock;
    auto last_time = clock::now();
    int frame_count = 0;

    while (true) {

        generate_signal(t_domain, 0.0f * float(frame), BLOCK_LEN*N_BLOCKS, frame);

        run_fft(t_domain, f_domain, BLOCK_LEN, N_BLOCKS);

        polchrome(f_domain, bitmap, BLOCK_LEN, N_BLOCKS);

        fft_postproc(f_domain, spec_bitmap, BLOCK_LEN, N_BLOCKS);


        launch_cuda_kernel(bitmap);
        drawGL(pbo, tex, WIDTH, HEIGHT);
        frame++;
        frame_count++;
        glutMainLoopEvent();

        // Print FPS every second
        auto now = clock::now();
        std::chrono::duration<double> elapsed = now - last_time;
        if (elapsed.count() >= 1.0) {
            printf("FPS: %d\n", frame_count);
            printf("Max rate: %f MHz\n", float(frame_count) * float(BLOCK_LEN) * float(N_BLOCKS) / 1e6f);
            frame_count = 0;
            last_time = now;
        }

        //usleep(1e6);
    }

    cleanup();
    return 0;
}