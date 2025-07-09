#pragma once
#include <thread>

extern float2 ping_buffer[];
extern float2 pong_buffer[];

void udp_init(int port = 2000);
void udp_close();
float2* process_next_buffer();
