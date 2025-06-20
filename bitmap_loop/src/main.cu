#include "gl_draw.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include "aux.h"
#include "bitmap.h"
#include "signal.h"

const int WIDTH = 2560;
const int HEIGHT = 1440;
const int SIGNAL_LENGTH = 2560;

cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;
int frame = 0; // Global frame counter


void launch_cuda_kernel(float2* d_signal) {
    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    fill_bitmap_spec<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame, d_signal);
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

    float2* d_signal = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_signal, SIGNAL_LENGTH * sizeof(float2)));

    while (true) {
        generate_signal(d_signal, 0.0f * float(frame), SIGNAL_LENGTH, frame);

        run_fft(d_signal, SIGNAL_LENGTH);


        launch_cuda_kernel(d_signal);
        drawGL(pbo, tex, WIDTH, HEIGHT);
        frame++;
        glutMainLoopEvent();

        //usleep(1e6);
    }

    cleanup();
    return 0;
}