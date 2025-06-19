#pragma once

#include <GL/glew.h>      // Always first!
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void initGLUT(int *argc, char **argv, void (*cleanupFunc)());
void initPixelBuffer(GLuint *pbo, GLuint *tex, cudaGraphicsResource **cuda_pbo_resource);
void drawGL(GLuint pbo, GLuint tex, int width, int height);

#ifdef __cplusplus
}
#endif