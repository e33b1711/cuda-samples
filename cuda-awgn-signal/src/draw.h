#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

// Forward declaration to avoid including CUDA headers in C++ files
struct cudaGraphicsResource;

// Initializes OpenGL view (projection, modelview, point size)
void setupGLView(float x_min, float x_max, float y_min, float y_max, float point_size);

// Handles the OpenGL draw loop and window events
void drawLoop(cudaGraphicsResource* cuda_vbo_resource, GLuint vbo, int signal_length, GLFWwindow* window);

// Cleans up OpenGL and CUDA resources
void cleanupDraw(cudaGraphicsResource* cuda_vbo_resource, GLuint vbo, GLFWwindow* window);

// This function handles all OpenGL/GLFW/CUDA interop and drawing
void displaySignalWithCudaInterop(float2* d_signal, int signal_length);