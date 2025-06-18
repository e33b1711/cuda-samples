cmake_minimum_required(VERSION 3.10)
project(cuda_awgn_signal LANGUAGES CXX CUDA)



set(CMAKE_CUDA_STANDARD 11)

# Specify the source files
set(SOURCES
    src/main.cu
    src/signal.cu
    src/utils.cu    # ...existing code...
    src/draw.cpp
)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE src)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
pkg_search_module(GLEW REQUIRED glew)

find_package(CUDA REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS} 
    src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLEW_LIBRARIES}
    cuda
    cudart
)

