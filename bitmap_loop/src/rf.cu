#include <hackrf.h>
#include <stdio.h>
#include <unistd.h>

int read_hackrf_samples(uint8_t *buffer, int num_samples, uint64_t freq_hz, uint32_t sample_rate)
{
    printf("Hello\n");
    hackrf_device *device = nullptr;
    int result = hackrf_init();
    if (result != HACKRF_SUCCESS)
    {
        fprintf(stderr, "hackrf_init() failed: %s\n", hackrf_error_name((hackrf_error)result));
        return -1;
    }

    printf("Hello0\n");
    result = hackrf_open(&device);
    if (result != HACKRF_SUCCESS)
    {
        fprintf(stderr, "hackrf_open() failed: %s\n", hackrf_error_name((hackrf_error)result));
        hackrf_exit();
        return -1;
    }

    printf("Hello1\n");
    hackrf_set_freq(device, freq_hz);
    printf("Hello2\n");
    hackrf_set_sample_rate(device, sample_rate);

    int received = 0;
    auto rx_callback = [](hackrf_transfer *transfer) -> int
    {
        uint8_t *buf = (uint8_t *)transfer->buffer;
        int to_copy = transfer->valid_length;
        memcpy(transfer->rx_ctx, buf, to_copy);
        return 0;
    };

    printf("Hello3\n");
    result = hackrf_start_rx(device, rx_callback, buffer);
    if (result != HACKRF_SUCCESS)
    {
        fprintf(stderr, "hackrf_start_rx() failed: %s\n", hackrf_error_name((hackrf_error)result));
        hackrf_close(device);
        hackrf_exit();
        return -1;
    }
    printf("Hello4\n");

    // Wait or implement a mechanism to collect num_samples
    // For demo, sleep for a short time

    sleep(2);

    printf("Hello5\n");
    result = hackrf_stop_rx(device);
    if (result != HACKRF_SUCCESS)
    {
        fprintf(stderr, "hackrf_start_rx() failed: %s\n", hackrf_error_name((hackrf_error)result));
        hackrf_close(device);
        hackrf_exit();
        return -1;
    }
    printf("Hello5\n");
    hackrf_close(device);
    hackrf_exit();
    return 0;
}