#include "gl_draw.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

#include "aux.h"


cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo = 0, tex = 0;


void initGLUT(int *argc, char **argv, void (*cleanupFunc)(), int height, int width) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Bitmap via GLUT");
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(err));
        exit(1);
    }
    atexit(cleanupFunc);
}


void initPixelBuffer(GLuint *pbo, GLuint *tex, cudaGraphicsResource **cuda_pbo_resource, int height, int width) {
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, 0, GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(cuda_pbo_resource, *pbo, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);
}


void drawGL(GLuint pbo, GLuint tex, int width, int height) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();

    glutSwapBuffers();
}


void dummy_display() {}


void draw_cleanup() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
}


void draw_init(const int height, const int width, int argc, char **argv){
    initGLUT(&argc, argv, draw_cleanup, height, width);
    glutDisplayFunc(dummy_display); // Register dummy display callback
    initPixelBuffer(&pbo, &tex, &cuda_pbo_resource, height, width);

}


void draw_loop(uchar4* bitmap, const int width, const int height){

    uchar4 *dptr;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource));
    CUDA_SAFE_CALL(cudaMemcpy(dptr, bitmap, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice););
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource));
    drawGL(pbo, tex, width, height);
    
    glutMainLoopEvent();
}