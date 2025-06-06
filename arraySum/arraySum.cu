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

 void array_sum_cpu(const int* a, const int* b, const int* c, int* result, const int num_elements){
    for (int gid = 0; gid < num_elements; gid++) {
       result[gid] = a[gid] + b[gid] + c[gid];
    }
 }

bool compare_result(const int* a, const int* b, const int num_elements){
    for (int gid = 0; gid < num_elements; gid++) {
       if (a[gid] != b[gid]) return false;
    }
    return true;
 }

/**
 * Host main routine
 */
int main(void)
{

    //constants
    const int num_bs = 6;
    const int block_sizes[num_bs] = {64, 64, 64, 128, 256, 512};


    //time stuff
    clock_t start, stop;
    double calc_cpu, host2gpu, gpu2host;
    double calc_gpu;


    //initalizing input data
    int num_elements = 1<<22;
    cout << "Number of alements: " << num_elements << endl;
    int *h_a = (int *)malloc(num_elements*sizeof(int));
    int *h_b = (int *)malloc(num_elements*sizeof(int));
    int *h_c = (int *)malloc(num_elements*sizeof(int));
    int *h_result_cpu = (int *)malloc(num_elements*sizeof(int));
    int *h_result_gpu = (int *)malloc(num_elements*sizeof(int));

    for (int i = 0; i < num_elements; i++) {
        h_a[i] = rand() / (int)RAND_MAX;
        h_b[i] = rand() / (int)RAND_MAX;
        h_c[i] = rand() / (int)RAND_MAX;
    }


    //alloc gpu mem
    int *d_a = NULL;
    int *d_b = NULL;
    int *d_c = NULL;
    int *d_result = NULL;
    gpuErrchk(cudaMalloc((void **)&d_a, num_elements*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_b, num_elements*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_c, num_elements*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_result, num_elements*sizeof(int)));


    //copy to gpu
    start = clock();
    gpuErrchk(cudaMemcpy(d_a, h_a, num_elements*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, num_elements*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c, h_c, num_elements*sizeof(int), cudaMemcpyHostToDevice));
    stop = clock();
    host2gpu = (double)(stop - start) / CLOCKS_PER_SEC;


    //run cpu reference
    start = clock();
    array_sum_cpu(h_a, h_b, h_c, h_result_cpu, num_elements);
    stop = clock();
    calc_cpu = (double)(stop - start) / CLOCKS_PER_SEC;


    //print cpu ref
    cout << "CPU               : " << calc_cpu << endl;


    //loop over block sizes
    for(int i_bs = 0; i_bs < num_bs; i_bs++){

        //loop parameter
        int block_size = block_sizes[i_bs];
        int grid_size = (num_elements / block_size) +1;
        dim3 dimGrid(grid_size, 1, 1);
        dim3 dimBlock(block_size, 1, 1);
        cout << "Grid size: " << grid_size << " block size: " << block_size << endl;

        //execute kernel
        start = clock();
        array_sum_gpu<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, d_result, num_elements);
        gpuErrchk(cudaDeviceSynchronize());
        stop = clock();
        calc_gpu = (double)(stop - start) / CLOCKS_PER_SEC;

        //device to host copy
        start = clock();
        gpuErrchk(cudaMemcpy(h_result_gpu, d_result, num_elements*sizeof(int), cudaMemcpyDeviceToHost));
        stop = clock();
        gpu2host = (double)(stop - start) / CLOCKS_PER_SEC;;

        //validate result
        assert(compare_result(h_result_gpu, h_result_cpu, num_elements) && "Result did not validate!");


        //print
        cout << "GPU:                " <<  (calc_gpu + gpu2host + host2gpu) << endl;
        cout << "GPU (calc only):    " <<  calc_gpu << endl;

    }

    return EXIT_SUCCESS;
}
