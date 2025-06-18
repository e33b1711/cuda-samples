#include <iostream>
#include <curand_kernel.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "signal.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

const int SIGNAL_LENGTH = 1024;

__global__ void generateComplexSignal(float2* signal, int length, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        signal[idx].x = curand_normal(&state); // Real part
        signal[idx].y = curand_normal(&state); // Imaginary part
    }
}

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

void drawLoop(cudaGraphicsResource* cuda_vbo_resource, GLuint vbo) {
    while (!glfwWindowShouldClose(glfwGetCurrentContext())) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw points
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, 0);
        glDrawArrays(GL_POINTS, 0, SIGNAL_LENGTH);
        glDisableClientState(GL_VERTEX_ARRAY);

        glfwSwapBuffers(glfwGetCurrentContext());
        glfwPollEvents();
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA Complex Signal", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    // Create OpenGL VBO
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, SIGNAL_LENGTH * sizeof(float2), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsResource* cuda_vbo_resource;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    // Map VBO for CUDA
    float2* d_signal = nullptr;
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_signal, &num_bytes, cuda_vbo_resource));

    // Generate signal and add noise
    int blockSize = 256;
    int numBlocks = (SIGNAL_LENGTH + blockSize - 1) / blockSize;
    generateComplexSignal<<<numBlocks, blockSize>>>(d_signal, SIGNAL_LENGTH, 1234ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float noiseVariance = 0.1f;
    addAWGN<<<numBlocks, blockSize>>>(d_signal, noiseVariance, SIGNAL_LENGTH, 1234ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    // Set up OpenGL view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-4, 4, -4, 4, -1, 1); // Adjust axis as needed
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPointSize(2.0f);

    // Draw loop
    drawLoop(cuda_vbo_resource, vbo);

    // Cleanup
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    glDeleteBuffers(1, &vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}