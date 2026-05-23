#include "hawkes.metal"

using namespace metal;

#define MARKOV_MI_PARTIAL_KERNEL(name, storage, scalar) \
    markov_mutual_information_partial_kernel<storage, scalar>( \
#define HAWKES_MARKOV_FINALIZE_KERNEL(name, storage, scalar) \
    hawkes_markov_finalize_kernel<storage, scalar>(scratch, out, reduction, partialCount, threadIndex); \
#define MARKOV_PARTITION_KERNEL(name, storage, scalar) \
    markov_blanket_partition_kernel<storage, scalar>( \
#define MARKOV_FLOW_KERNEL(name, storage, scalar) \
    markov_flow_kernel<storage, scalar>( \
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_float32_partial, Float32HawkesMarkovStorage, float)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float32, Float32HawkesMarkovStorage, float)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_float32, Float32HawkesMarkovStorage, float)
MARKOV_FLOW_KERNEL(markov_flow_float32, Float32HawkesMarkovStorage, float)
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_float16_partial, Float16HawkesMarkovStorage, half)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float16, Float16HawkesMarkovStorage, half)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_float16, Float16HawkesMarkovStorage, half)
MARKOV_FLOW_KERNEL(markov_flow_float16, Float16HawkesMarkovStorage, half)
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_bfloat16_partial, BFloat16HawkesMarkovStorage, ushort)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_bfloat16, BFloat16HawkesMarkovStorage, ushort)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_bfloat16, BFloat16HawkesMarkovStorage, ushort)
MARKOV_FLOW_KERNEL(markov_flow_bfloat16, BFloat16HawkesMarkovStorage, ushort)
