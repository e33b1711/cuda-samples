cmake_minimum_required(VERSION 3.18)
project(cuda_glut_bitmap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    src
)

set(SOURCES
    src/aux.cu
    src/gl_draw.cu
    src/fft.cu
    src/signal.cu
    src/main.cu
    src/polycrome.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(cuda_glut_bitmap
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    GL
    GLU
    glut
    cuda
    cudart
    cufft
)

set_target_properties(cuda_glut_bitmap PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)