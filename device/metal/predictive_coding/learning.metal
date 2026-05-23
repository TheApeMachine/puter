#include "predictive_coding.metal"

using namespace metal;

#define RESEARCH_BINARY_KERNEL(name, body, storage, scalar) \
#define RESEARCH_UNARY_KERNEL(name, body, storage, scalar) \
#define PC_PREDICTION_KERNEL(name, storage, scalar) \
#define PC_UPDATE_REPRESENTATION_KERNEL(name, storage, scalar) \
#define PC_UPDATE_WEIGHTS_KERNEL(name, storage, scalar) \
static inline void pc_update_representation_kernel(
static inline void pc_update_weights_kernel(
    pc_update_representation_kernel<storage, scalar>( \
    pc_update_weights_kernel<storage, scalar>( \
PC_UPDATE_REPRESENTATION_KERNEL(pc_update_representation_float32, Float32ResearchStorage, float)
PC_UPDATE_WEIGHTS_KERNEL(pc_update_weights_float32, Float32ResearchStorage, float)
PC_UPDATE_REPRESENTATION_KERNEL(pc_update_representation_float16, Float16ResearchStorage, half)
PC_UPDATE_WEIGHTS_KERNEL(pc_update_weights_float16, Float16ResearchStorage, half)
PC_UPDATE_REPRESENTATION_KERNEL(pc_update_representation_bfloat16, BFloat16ResearchStorage, ushort)
PC_UPDATE_WEIGHTS_KERNEL(pc_update_weights_bfloat16, BFloat16ResearchStorage, ushort)
