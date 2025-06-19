#include "gl_draw.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#define WIDTH  2560
#define HEIGHT 1440
#define SIGNAL_LENGTH 2560

cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;
int frame = 0; // Global frame counter

// CUDA kernel to fill the bitmap with a gradient
__global__ void fill_bitmap_kernel(uchar4 *ptr, int width, int height, int frame, float2* d_signal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    ptr[idx].x = 0;
    ptr[idx].y = 0;
    ptr[idx].z = 0;
    ptr[idx].w = 255; // A
    int y_real = int( 200.0f * d_signal[x].x + 700.0f);
    int y_imag = int( 200.0f * d_signal[x].y + 700.0f);
    //int y_signal = 34;
    if (y_real == y){
        ptr[idx].y = 255;
    }
    if (y_imag == y){
        ptr[idx].x = 255;
    }
}

#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

void launch_cuda_kernel(float2* d_signal) {
    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    fill_bitmap_kernel<<<grid, block>>>(dptr, WIDTH, HEIGHT, frame, d_signal);
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

__global__ void addAWGN(float2* signal, float noiseVariance, int length, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float noiseReal = curand_normal(&state) * sqrt(noiseVariance);
        float noiseImag = curand_normal(&state) * sqrt(noiseVariance);
        signal[idx].x += noiseReal;
        signal[idx].y += noiseImag;
    }
}

__global__ void generatePhasorSignal(float2* signal, int length, float omega, float phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        float angle = omega * idx + phi;
        signal[idx].x = cosf(angle); // Real part
        signal[idx].y = sinf(angle); // Imaginary part
    }
}


void generate_signal(float2* d_signal, float phi){
    int blockSize = 256;
    float omega = 2.0f * 3.14159265359f * 5.0f / SIGNAL_LENGTH; // 5 cycles over the signal
    int numBlocks = (SIGNAL_LENGTH + blockSize - 1) / blockSize;
    generatePhasorSignal<<<numBlocks, blockSize>>>(d_signal, SIGNAL_LENGTH, omega, phi);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    float noiseVariance = 0.01f;
    addAWGN<<<numBlocks, blockSize>>>(d_signal, noiseVariance, SIGNAL_LENGTH, (unsigned long long) frame);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


int main(int argc, char **argv) {
    initGLUT(&argc, argv, cleanup, HEIGHT, WIDTH);
    glutDisplayFunc(dummy_display); // Register dummy display callback
    initPixelBuffer(&pbo, &tex, &cuda_pbo_resource, HEIGHT, WIDTH);

    struct timeval start, now;
    gettimeofday(&start, NULL);
    int frames = 0;

    float2* d_signal = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_signal, SIGNAL_LENGTH * sizeof(float2)));

    while (true) {
        generate_signal(d_signal, 0.0f * float(frame));

        launch_cuda_kernel(d_signal);
        drawGL(pbo, tex, WIDTH, HEIGHT);
        frame++;
        frames++;
        glutMainLoopEvent();

        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
        if (elapsed >= 1.0) {
            printf("FPS: %d\n", frames);
            frames = 0;
            start = now;
        }
        //usleep(160000);
    }

    cleanup();
    return 0;
}