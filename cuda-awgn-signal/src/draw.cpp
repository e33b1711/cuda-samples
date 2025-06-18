#include "draw.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>

void displaySignalWithCudaInterop(float2* d_signal, int signal_length) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* window = glfwCreateWindow(1600, 1200, "CUDA Complex Signal", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }

    // Create OpenGL VBO
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, signal_length * sizeof(float2), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Map VBO for CUDA and copy data
    float2* vbo_ptr = nullptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&vbo_ptr, &num_bytes, cuda_vbo_resource);
    cudaMemcpy(vbo_ptr, d_signal, signal_length * sizeof(float2), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Set up OpenGL view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-4, 4, -4, 4, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPointSize(2.0f);

    // Draw loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, 0);
        glDrawArrays(GL_POINTS, 0, signal_length);
        glDisableClientState(GL_VERTEX_ARRAY);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
}