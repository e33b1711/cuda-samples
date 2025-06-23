#include <stdio.h>
#include <chrono>
#include <unistd.h>



void time_info(const int block_len, const int n_blocks){
    using clock = std::chrono::high_resolution_clock;
    static auto last_time = clock::now();
    static int frame_count = 0;
    auto now = clock::now();
    std::chrono::duration<double> elapsed = now - last_time;
    if (elapsed.count() >= 1.0) {
        printf("FPS: %d\n", frame_count);
        printf("Max rate: %f MHz\n", float(frame_count) * float(block_len) * float(n_blocks) / 1e6f);
        frame_count = 0;
        last_time = now;
    }
    frame_count++;
}

