cmake_minimum_required(VERSION 3.10)
project(cuda_awgn_signal LANGUAGES CXX CUDA)



set(CMAKE_CUDA_STANDARD 11)

# Specify the source files
set(SOURCES
    src/main.cu
    src/signal.cu
    src/utils.cu
)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE src)