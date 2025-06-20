#include "gl_draw.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono> // Add at the top
#include <unistd.h>

#include "aux.h"
#include "bitmap.h"
#include "signal.h"

const int WIDTH = 1024;
const int HEIGHT = 512;
const int SIGNAL_LENGTH = 1024;
const int COUNT = 1024*16;

cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;
int frame = 0; // Global frame counter


void launch_cuda_kernel(float* mean, float* min, float* max) {
    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    fill_bitmap_spec<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame, mean, 0, true);
    fill_bitmap_spec<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame, min, 1, false);
    fill_bitmap_spec<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame, max, 2, false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
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
    float* f_max = nullptr;
    float* f_min = nullptr;
    float* f_mean = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&t_domain, SIGNAL_LENGTH * COUNT * sizeof(float2)));
    CUDA_SAFE_CALL(cudaMalloc(&f_domain, SIGNAL_LENGTH * COUNT * sizeof(float2)));

    CUDA_SAFE_CALL(cudaMalloc(&f_max, SIGNAL_LENGTH * COUNT * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&f_min, SIGNAL_LENGTH * COUNT * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&f_mean, SIGNAL_LENGTH * COUNT * sizeof(float)));

    using clock = std::chrono::high_resolution_clock;
    auto last_time = clock::now();
    int frame_count = 0;

    while (true) {

        generate_signal(t_domain, 0.0f * float(frame), SIGNAL_LENGTH*COUNT, frame);

        run_fft(t_domain, f_domain, SIGNAL_LENGTH, COUNT);

        fft_postproc(f_domain, f_max, f_min, f_mean, SIGNAL_LENGTH, COUNT);


        launch_cuda_kernel(f_max, f_min, f_mean);
        drawGL(pbo, tex, WIDTH, HEIGHT);
        frame++;
        frame_count++;
        glutMainLoopEvent();

        // Print FPS every second
        auto now = clock::now();
        std::chrono::duration<double> elapsed = now - last_time;
        if (elapsed.count() >= 1.0) {
            printf("FPS: %d\n", frame_count);
            frame_count = 0;
            last_time = now;
        }

        //usleep(1e6);
    }

    cleanup();
    return 0;
}