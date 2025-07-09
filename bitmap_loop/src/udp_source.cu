#include <iostream>
#include <thread>
#include <atomic>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <condition_variable>
#include <mutex>
#include <cuda_runtime.h>
#include "aux.h"

std::atomic<bool> running(true);

constexpr size_t BUF_SIZE = 1024 * 256 ;
float2* ping_buffer = nullptr;
float2* pong_buffer = nullptr;
size_t ping_offset = 0;
size_t pong_offset = 0;
std::atomic<bool> ping_ready(false);
std::atomic<bool> pong_ready(false);
std::mutex buf_mutex;
std::condition_variable buf_cv;

void udp_receive_thread(int sockfd) {
    sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    bool use_ping = true;
    while (running.load()) {
        float2* target = use_ping ? ping_buffer : pong_buffer;
        size_t& offset = use_ping ? ping_offset : pong_offset;
        ssize_t n = recvfrom(sockfd, reinterpret_cast<char*>(target) + offset * sizeof(float2), (BUF_SIZE - offset) * sizeof(float2), 0, (sockaddr*)&client_addr, &addr_len);
        if (n > 0) {
            offset += n / sizeof(float2);
            if (offset >= BUF_SIZE) {
                // Buffer is full, mark as ready and switch
                if (use_ping) {
                    ping_ready = true;
                } else {
                    pong_ready = true;
                }
                buf_cv.notify_one();
                use_ping = !use_ping;
                // Reset offset for new buffer
                (use_ping ? ping_offset : pong_offset) = 0;
            }
        }
    }
}

int sockfd = -1;
std::thread recv_thread;
bool local_use_ping = true;

void udp_init(int port = 2000) {
    // Allocate pinned host memory for ping and pong buffers
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&ping_buffer, BUF_SIZE * sizeof(float2), cudaHostAllocMapped));
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&pong_buffer, BUF_SIZE * sizeof(float2), cudaHostAllocMapped));

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        exit(1);
    }
    sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);
    if (bind(sockfd, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        ::close(sockfd);
        exit(1);
    }
    recv_thread = std::thread(udp_receive_thread, sockfd);
    std::cout << "UDP server listening on port " << port << std::endl;
}

void udp_close() {
    running = false;
    buf_cv.notify_all();
    if (recv_thread.joinable()) recv_thread.join();
    if (sockfd >= 0) ::close(sockfd);
    if (ping_buffer) cudaFreeHost(ping_buffer);
    if (pong_buffer) cudaFreeHost(pong_buffer);
}

float2* process_next_buffer() {
    std::unique_lock<std::mutex> lock(buf_mutex);
    buf_cv.wait(lock, [&]{ return (local_use_ping ? ping_ready.load() : pong_ready.load()) || !running.load(); });
    if (!running.load()) return nullptr;
    float2* src = local_use_ping ? ping_buffer : pong_buffer;
    // Process data in src buffer (always BUF_SIZE float2s)
    //std::cout << "Processing full buffer (" << (local_use_ping ? "ping" : "pong") << ")" << std::endl;
    // Reset ready flag
    if (local_use_ping) ping_ready = false; else pong_ready = false;
    local_use_ping = !local_use_ping;
    return src;
}

// Remove main() from this file, now in main.cpp
