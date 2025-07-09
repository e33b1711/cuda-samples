#pragma once
#include <thread>
#include <cuda_runtime.h>

extern char ping_buffer[];
extern char pong_buffer[];

void udp_init(int port = 2000);
void udp_close();
float2* process_next_buffer();
