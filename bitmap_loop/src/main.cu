#include "gl_draw.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <assert.h>

#define WIDTH  512
#define HEIGHT 512

cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;
int frame = 0; // Global frame counter

// CUDA kernel to fill the bitmap with a gradient
__global__ void fill_bitmap_kernel(uchar4 *ptr, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    ptr[idx].x = (x + frame) % 255; // Animate R
    ptr[idx].y = (y + frame) % 255; // Animate G
    ptr[idx].z = (frame * 2) % 255; // Animate B
    ptr[idx].w = 255; // A
}

#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

void launch_cuda_kernel() {
    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    fill_bitmap_kernel<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame);
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
    initGLUT(&argc, argv, cleanup);
    glutDisplayFunc(dummy_display); // Register dummy display callback
    initPixelBuffer(&pbo, &tex, &cuda_pbo_resource);

    while (true) {
        launch_cuda_kernel();
        drawGL(pbo, tex, WIDTH, HEIGHT);
        frame++;
        glutMainLoopEvent();
        // usleep(16000);
    }

    cleanup();
    return 0;
}