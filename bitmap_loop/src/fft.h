#pragma once
#include <cuda_runtime.h>

#include "params.h"

void run_fft(const context ctx, const ps params);
void fft_postproc(const context ctx, const ps params);
