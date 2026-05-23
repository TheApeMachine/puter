#include "hawkes.metal"

using namespace metal;

#define HAWKES_INTENSITY_KERNEL(name, storage, scalar) \
    hawkes_intensity_kernel<storage, scalar>( \
HAWKES_INTENSITY_KERNEL(hawkes_intensity_float32, Float32HawkesMarkovStorage, float)
HAWKES_INTENSITY_KERNEL(hawkes_intensity_float16, Float16HawkesMarkovStorage, half)
HAWKES_INTENSITY_KERNEL(hawkes_intensity_bfloat16, BFloat16HawkesMarkovStorage, ushort)
