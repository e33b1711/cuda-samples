#pragma once
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cuda_runtime.h>

class UdpSource {
  public:
    UdpSource(size_t buf_size = 1024 * 256);
    ~UdpSource();
    void init(int port = 2000);
    void close();
    float2* process_next_buffer();
    size_t get_buf_size() const;
  private:
    void udp_receive_thread();
    const size_t BUF_SIZE;
    float2* ping_buffer;
    float2* pong_buffer;
    size_t ping_offset;
    size_t pong_offset;
    std::atomic<bool> ping_ready;
    std::atomic<bool> pong_ready;
    std::mutex buf_mutex;
    std::condition_variable buf_cv;
    std::atomic<bool> running;
    int sockfd;
    std::thread recv_thread;
    bool local_use_ping;
};
