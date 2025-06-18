#ifndef SIGNAL_H
#define SIGNAL_H

#include <cuComplex.h>

void generateComplexSignal(cuFloatComplex* signal, int numSamples, float frequency, float amplitude);
void addAWGN(cuFloatComplex* signal, int numSamples, float noisePower);

#endif // SIGNAL_H