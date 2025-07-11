#include "udp_source.h"
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


UdpSource::UdpSource(size_t buf_size)
    : BUF_SIZE(buf_size), ping_buffer(nullptr), pong_buffer(nullptr),
      ping_offset(0), pong_offset(0), ping_ready(false), pong_ready(false),
      running(true), sockfd(-1), local_use_ping(true) {}

UdpSource::~UdpSource() {
    close();
}

void UdpSource::init(int port) {
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
    recv_thread = std::thread(&UdpSource::udp_receive_thread, this);
    std::cout << "UDP server listening on port " << port << std::endl;
}

void UdpSource::close() {
    running = false;
    buf_cv.notify_all();
    if (recv_thread.joinable()) recv_thread.join();
    if (sockfd >= 0) ::close(sockfd);
    if (ping_buffer) cudaFreeHost(ping_buffer);
    if (pong_buffer) cudaFreeHost(pong_buffer);
    std::cout << "UDP server closed." << std::endl;
}

float2* UdpSource::process_next_buffer() {
    std::unique_lock<std::mutex> lock(buf_mutex);
    buf_cv.wait(lock, [&]{ return (local_use_ping ? ping_ready.load() : pong_ready.load()) || !running.load(); });
    if (!running.load()) return nullptr;
    float2* src = local_use_ping ? ping_buffer : pong_buffer;
    // Reset ready flag
    if (local_use_ping) ping_ready = false; else pong_ready = false;
    local_use_ping = !local_use_ping;
    return src;
}

size_t UdpSource::get_buf_size() const { return BUF_SIZE; }

void UdpSource::udp_receive_thread() {
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
                if (use_ping) {
                    ping_ready = true;
                } else {
                    pong_ready = true;
                }
                buf_cv.notify_one();
                use_ping = !use_ping;
                (use_ping ? ping_offset : pong_offset) = 0;
            }
        }
    }
}
