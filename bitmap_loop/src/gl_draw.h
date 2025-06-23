#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

void draw_init(const int height, const int width);
void draw_cleanup();
void draw_loop(uchar4* bitmap, const int width, const int height);