#include "predictive_coding.metal"

using namespace metal;

PC_PREDICTION_KERNEL(pc_prediction_float32, Float32ResearchStorage, float)
RESEARCH_BINARY_KERNEL(
    pc_prediction_error_float32,
    pc_prediction_error_kernel,
    Float32ResearchStorage,
    float
)
PC_PREDICTION_KERNEL(pc_prediction_float16, Float16ResearchStorage, half)
RESEARCH_BINARY_KERNEL(
    pc_prediction_error_float16,
    pc_prediction_error_kernel,
    Float16ResearchStorage,
    half
)
PC_PREDICTION_KERNEL(pc_prediction_bfloat16, BFloat16ResearchStorage, ushort)
RESEARCH_BINARY_KERNEL(
    pc_prediction_error_bfloat16,
    pc_prediction_error_kernel,
    BFloat16ResearchStorage,
    ushort
)
