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


    //i=0
    float mean = 0.0;
    int i = start;
    for (int offset = 0; offset<num_threads; offset++){
        int index = (i+buffer_len-offset) % buffer_len;
        mean += x[index];
    }
    xmean[i] = mean;


    for (int i = start+1; i < start + partial_buffer_len; i++) {
        int index = (i+buffer_len-num_threads) % buffer_len;
        mean += x[i];
        mean -= x[index];
        xmean[i] = mean;
    }
}


__global__ void pp_mean_gpu(const float* xmean, float* y, const int buffer_len, const int num_threads, const int average_len)
{
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int thread_in_block = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int block_num = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int thread_unique = block_num * block_size + thread_in_block;

    //todo initial values
    y[buffer_len-num_threads+thread_unique] = 0.0;


    for (int i = thread_unique; i < buffer_len; i += num_threads) {
        int other_x = (i+buffer_len-average_len) % buffer_len;
        int other_y = (i+buffer_len-num_threads) % buffer_len;
        y[i] =  y[other_y] + xmean[i] - xmean[other_x];
    }
}



void mean_cpu(const float* x, float* xmean, const int buffer_len, const int num_threads){

    for (int i = 0; i < buffer_len; i++) {
        float mean = 0.0;   
        for (int offset = 0; offset<num_threads; offset++){
            int index = (i+ buffer_len - offset) % buffer_len;
            mean += x[index];
        }   
        xmean[i] = mean;
    }
}

void pp_mean_cpu(const float* xmean, float* y, const int buffer_len, const int num_threads, const int average_len)
{

    for (int i = 0; i < buffer_len; i++) {
        int other_i = (i+buffer_len-average_len) % buffer_len;
        int other_y = (i+buffer_len-num_threads) % buffer_len;
        y[i] =  y[other_y] + xmean[i] - xmean[other_i];
    }
}

void moving_average(const float* x, float* y, const int average_len, const int buffer_len){

    for(int i = 0; i<buffer_len; i++){
        int other_i = (i+buffer_len-average_len) % buffer_len;
        y[i] =  y[i-1] + x[i] - x[other_i];
    }
}

bool compare(const float* a, const float* b, const int buffer_len){
    for(int i = 0; i<buffer_len; i++){
        if (abs(a[i] - b[i])>0.005) {
            cout << "compare error: " << i << " " << a[i] << " " << b[i] << endl;
            return false;
        }
    }
    return true;
}


int main(void)
{

    // parameters
    const int average_len = 2<<12;
    const int buffer_len = 2<<20;
    const int log2_num_threads = 7;
    const int num_threads = 2<<log2_num_threads;

    //const int average_len = 64;
    //const int buffer_len = 2<<10;
    //const int num_threads = 4;

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(num_threads, 1, 1);
    
    // host buffers
    float *h_x = (float *)malloc(buffer_len*sizeof(float));
    float *h_xmean = (float *)malloc(buffer_len*sizeof(float));
    float *h_y = (float *)malloc(buffer_len*sizeof(float));
    float *h_y_gold = (float *)malloc(buffer_len*sizeof(float));

    // input signal 
    for (int i = 0; i < buffer_len; i++) {
        if(i < buffer_len-average_len-num_threads){
            h_x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }else{
            h_x[i] = 0.0;
        }
    }

    // cpu ref calculation
    mean_cpu(h_x, h_xmean, buffer_len, num_threads);
    for(int i=0; i<num_threads; i++) h_y[buffer_len-i] = 0.0;
    pp_mean_cpu(h_xmean, h_y, buffer_len, num_threads, average_len);

    clock_t start = clock();
    h_y_gold[buffer_len-1] = 0.0;
    moving_average(h_x, h_y_gold, average_len, buffer_len);
    clock_t stop = clock();
    double calc_cpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "CPU: " << calc_cpu << endl;



    //alloc gpu mem
    float *d_x = NULL;
    float *d_xmean = NULL;
    float *d_y = NULL;
    gpuErrchk(cudaMalloc((void **)&d_x, buffer_len*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_xmean, buffer_len*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_y, buffer_len*sizeof(float)));


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
    cout << "GPU: " << calc_gpu << endl;

    start = clock();
    pp_mean_gpu<<<dimGrid, dimBlock>>>(d_xmean, d_y, buffer_len, num_threads, average_len);
    gpuErrchk(cudaDeviceSynchronize());
    stop = clock();
    calc_gpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "GPU: " << calc_gpu << endl;

    //device to host copy
    float *h_xmean_gpu = (float *)malloc(buffer_len*sizeof(float));
    float *h_y_gpu = (float *)malloc(buffer_len*sizeof(float));
    start = clock();
    gpuErrchk(cudaMemcpy(h_xmean_gpu, d_xmean, buffer_len*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_y_gpu, d_y, buffer_len*sizeof(float), cudaMemcpyDeviceToHost));
    stop = clock();
    double gpu2host = (double)(stop - start) / CLOCKS_PER_SEC;;
    cout << "gpu2host: " << gpu2host << endl;


    //
    if(buffer_len<=2<<10){
        dump(h_x, buffer_len, "h_x.bin");
        dump(h_xmean, buffer_len, "h_xmean.bin");
        dump(h_y, buffer_len, "h_y.bin");
        dump(h_y_gold, buffer_len, "h_y_gold.bin");
        dump(h_xmean_gpu, buffer_len, "h_xmean_gpu.bin");
        dump(h_y_gpu, buffer_len, "h_y_gpu.bin");
    }
    if (!compare(h_y, h_y_gold, buffer_len)) cout <<  "gold does not match cpu." << endl;
    if (!compare(h_xmean_gpu, h_xmean, buffer_len)) cout << "h_xmean errror" << endl;
    if (!compare(h_y_gpu, h_y, buffer_len)) cout << "< errror" << endl;

    return EXIT_SUCCESS;
}
