#include <stdio.h>
#include <iostream>
#include <cassert>
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

int dump(const float* array, size_t size, const std::string& filename) {
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



void moving_average(const float* x, float* y, const size_t start, const size_t average_len, const size_t buffer_len){

    for(size_t i = start; i<buffer_len; i++){
        size_t other_i = (i+buffer_len-average_len) % buffer_len;
        y[i] =  y[i-1] + x[i] - x[other_i];
    }
}


int main(void)
{

    // parameters
    const size_t average_len = 2<<8;
    const size_t buffer_len = 2<<16;
    
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
    moving_average(h_x, h_y, 1, buffer_len, average_len);
    clock_t stop = clock();
    double calc_cpu = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "CPU: " << calc_cpu << endl;

    dump(h_x, buffer_len, "h_x.bin");
    dump(h_y, buffer_len, "h_y.bin");

    return EXIT_SUCCESS;
}
