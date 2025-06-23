#include <stdio.h>
#include <chrono>
#include <unistd.h>


void time_info(){
    static auto last_time = clock::now();
    static int frame_count = 0;
    auto now = clock::now();
    std::chrono::duration<double> elapsed = now - last_time;
    if (elapsed.count() >= 1.0) {
        printf("FPS: %d\n", frame_count);
        printf("Max rate: %f MHz\n", float(frame_count) * float(BLOCK_LEN) * float(N_BLOCKS) / 1e6f);
        frame_count = 0;
        last_time = now;
    }
    frame_count++;
}