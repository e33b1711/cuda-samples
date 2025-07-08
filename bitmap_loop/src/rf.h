#pragma once
#include <hackrf.h>
#include <stdio.h>

int read_hackrf_samples(uint8_t* buffer, int num_samples, uint64_t freq_hz, uint32_t sample_rate);