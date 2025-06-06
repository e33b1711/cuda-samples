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


int dump(const float* array, int size, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file!" << std::endl;
        return -1;
    }
    file.write(reinterpret_cast<const char*>(array), size * sizeof(float));
    if (!file) {
        std::cerr << "Failed to write data!" << std::endl;
        return -1;
    }
    return 0;
}


__global__ void mean_gpu(const float* x, float* xmean, const int buffer_len, const int num_threads)
{
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int thread_in_block = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int block_num = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int thread_unique = block_num * block_size + thread_in_block;

    int partial_buffer_len = buffer_len / num_threads;
    int start = thread_unique * partial_buffer_len;

    printf(" thread_unique: %d partial_buffer_len %d start %d \n", thread_unique, partial_buffer_len, start);


    //TODO this can be done recursivly
    for (int i = start; i < start + partial_buffer_len; i++) {
        float mean = 0.0;
        if (thread_unique==0) printf("start: %d | ", i);     
        for (int offset = 0; offset<num_threads; offset++){
            int index = (i+offset) % buffer_len;
            mean += x[index];
            if (thread_unique==0) printf("%d ", index);
        }
        if (thread_unique==0) printf("\n");     
        xmean[i] = mean;
    }
}



void moving_average(const float* x, float* y, const int start, const int average_len, const int buffer_len){

    for(int i = start; i<buffer_len; i++){
        int other_i = (i+buffer_len-average_len) % buffer_len;
        y[i] =  y[i-1] + x[i] - x[other_i];
    }
}


int main(void)
{

    // parameters
    const int average_len = 16;
    const int buffer_len = 256;
    const int num_threads = 4;

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(num_threads, 1, 1);
    
    // host buffers
    float *h_x = (float *)malloc(buffer_len*sizeof(float));
    float *h_y = (float *)malloc(buffer_len*sizeof(float));

    // input signal 
    for (int i = 0; i < buffer_len; i++) {
        if(i < buffer_len-average_len){
            h_x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }else{
            h_x[i] = 0.0;
        }
    }

    // cpu ref calculation
    clock_t start = clock();
    h_y[0] = 0.0;
    moving_average(h_x, h_y, 1, average_len, buffer_len);
    clock_t stop = clock();
    double calc_cpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "CPU: " << calc_cpu << endl;



    //alloc gpu mem
    float *d_x = NULL;
    float *d_xmean = NULL;
    gpuErrchk(cudaMalloc((void **)&d_x, buffer_len*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_xmean, buffer_len*sizeof(float)));

    
    //copy to gpu
    start = clock();
    gpuErrchk(cudaMemcpy(d_x, h_x, buffer_len*sizeof(float), cudaMemcpyHostToDevice));
    stop = clock();
    double host2gpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "host2gpu: " << host2gpu << endl;

    start = clock();
    mean_gpu<<<dimGrid, dimBlock>>>(d_x, d_xmean, buffer_len, num_threads);
    gpuErrchk(cudaDeviceSynchronize());
    stop = clock();
    double calc_gpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "GPU: " << calc_cpu << endl;

    //device to host copy
    float *h_xmean = (float *)malloc(buffer_len*sizeof(float));
    start = clock();
    gpuErrchk(cudaMemcpy(h_xmean, d_xmean, buffer_len*sizeof(float), cudaMemcpyDeviceToHost));
    stop = clock();
    double gpu2host = (double)(stop - start) / CLOCKS_PER_SEC;;
    cout << "gpu2host: " << gpu2host << endl;


    // 
    dump(h_x, buffer_len, "h_x.bin");
    dump(h_y, buffer_len, "h_y.bin");
    dump(h_xmean, buffer_len, "h_xmean.bin");

    return EXIT_SUCCESS;
}
