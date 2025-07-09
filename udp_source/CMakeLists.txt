cmake_minimum_required(VERSION 3.18)
project(cuda_glut_bitmap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

set(CMAKE_CXX_STANDARD 14)


find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}cmake_minimum_required(VERSION 3.18)
project(cuda_glut_bitmap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS}
    src
)

set(SOURCES
    src/main.cu
    src/udp_source.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(cuda_glut_bitmap
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OpenCV_LIBS}
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
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS}
    src
)

add_library(udp_source_lib
    src/udp_source.cu
)

target_include_directories(udp_source_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(udp_source src/main.cpp)
target_link_libraries(udp_source udp_source_lib pthread)
