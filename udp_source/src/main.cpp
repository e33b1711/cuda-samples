#include "udp_source.h"
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    udp_init();
    float2* src;
    float2* dest = (float2*) malloc(2024* sizeof(float2));

    for(int i=0; i< 200; i++) {
        src = process_next_buffer();
        memcpy(dest, src, 20484* sizeof(float2));
    }
    udp_close();
    return 0;
}
