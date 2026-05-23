#include "predictive_coding.metal"

using namespace metal;

#define RESEARCH_BINARY_KERNEL(name, body, storage, scalar) \
#define RESEARCH_UNARY_KERNEL(name, body, storage, scalar) \
#define PC_PREDICTION_KERNEL(name, storage, scalar) \
#define PC_UPDATE_REPRESENTATION_KERNEL(name, storage, scalar) \
#define PC_UPDATE_WEIGHTS_KERNEL(name, storage, scalar) \
static inline void pc_prediction_kernel(
static inline void pc_prediction_error_kernel(
    pc_prediction_kernel<storage, scalar>(weights, state, out, inCount, outIndex); \
PC_PREDICTION_KERNEL(pc_prediction_float32, Float32ResearchStorage, float)
    pc_prediction_error_float32,
    pc_prediction_error_kernel,
PC_PREDICTION_KERNEL(pc_prediction_float16, Float16ResearchStorage, half)
    pc_prediction_error_float16,
    pc_prediction_error_kernel,
PC_PREDICTION_KERNEL(pc_prediction_bfloat16, BFloat16ResearchStorage, ushort)
    pc_prediction_error_bfloat16,
    pc_prediction_error_kernel,
