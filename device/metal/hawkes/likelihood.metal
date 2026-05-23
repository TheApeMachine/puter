#include "hawkes.metal"

using namespace metal;

#define HAWKES_LOG_PARTIAL_KERNEL(name, storage, scalar) \
    hawkes_log_likelihood_partial_kernel<storage, scalar>( \
#define HAWKES_LOG_FINALIZE_KERNEL(name, storage, scalar) \
    hawkes_log_likelihood_finalize_kernel<storage, scalar>( \
#define HAWKES_MARKOV_FINALIZE_KERNEL(name, storage, scalar) \
    hawkes_markov_finalize_kernel<storage, scalar>(scratch, out, reduction, partialCount, threadIndex); \
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_float32_partial, Float32HawkesMarkovStorage, float)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_float32_finalize, Float32HawkesMarkovStorage, float)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float32, Float32HawkesMarkovStorage, float)
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_float16_partial, Float16HawkesMarkovStorage, half)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_float16_finalize, Float16HawkesMarkovStorage, half)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float16, Float16HawkesMarkovStorage, half)
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_bfloat16_partial, BFloat16HawkesMarkovStorage, ushort)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_bfloat16_finalize, BFloat16HawkesMarkovStorage, ushort)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_bfloat16, BFloat16HawkesMarkovStorage, ushort)
