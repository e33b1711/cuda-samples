#include <stdio.h>
#include <iostream>
#include <cassert>
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const string file, int line, bool abort = true){
    if (code != cudaSuccess){
        cerr <<  "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
        if (abort) exit(code);
    }
}

__global__ void array_sum_gpu(const int* a, const int* b, const int* c, int* result, const int num_elements)
{
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int thread_in_block = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int block_num = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int gid = block_num * block_size + thread_in_block;

    //printf(" array_sum_gpu, gid: %d \n", gid);
    if (gid < num_elements){
        result[gid] = a[gid] + b[gid] + c[gid];
    }
}

 void moving_average(const float* x, const float* y, const size_t offset, const size_t num_samples, const size_t average_len, const size_t buffer_len){


    for(size_t i = offset; i<offset+num_samples; i++){
        if !(i < buffer_len) break;
        size_t other_i = (i+buffer_len-average_len) % buffer_len;
        y[i] =  y[i-1] + x[i] - x[other_i];
    }
}

/**
 * Host main routine
 */
int main(void)
{


    const size_t average_len = 2<<16;
    const size_t buffer_len = 2<<28;
    const size_t partial_buf_len = 2<<24;
    int *h_x = (float *)malloc(num_elements*sizeof(float));
    int *h_y = (float *)malloc(num_elements*sizeof(float));

    for (int i = 0; i < buffer_len; i++) {
        h_x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    //starting the moving average
    h_y[0] = h_x[0];
    for (int i = 1; i < average_len; i++) {
        h_y[i] =  h_y[i-1] + h_x[i]
    }
    size_t offset = average_len;

    //cpu calculating the rest
    moving_average(h_x, h_y, offset, buffer_len, average_len, buffer_len){



    return EXIT_SUCCESS;
}
