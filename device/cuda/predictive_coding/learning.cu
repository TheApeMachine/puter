#include "predictive_coding.cuh"

PC_UPDATE_REPRESENTATION_KERNEL(
    pc_update_representation_float32,
    pc_load_f32,
    pc_store_f32,
    float
)
PC_UPDATE_WEIGHTS_KERNEL(
    pc_update_weights_float32,
    pc_load_f32,
    pc_store_f32,
    float
)

PC_UPDATE_REPRESENTATION_KERNEL(
    pc_update_representation_float16,
    pc_load_f16,
    pc_store_f16,
    __half
)
PC_UPDATE_WEIGHTS_KERNEL(
    pc_update_weights_float16,
    pc_load_f16,
    pc_store_f16,
    __half
)

PC_UPDATE_REPRESENTATION_KERNEL(
    pc_update_representation_bfloat16,
    pc_load_bf16,
    pc_store_bf16,
    __nv_bfloat16
)
PC_UPDATE_WEIGHTS_KERNEL(
    pc_update_weights_bfloat16,
    pc_load_bf16,
    pc_store_bf16,
    __nv_bfloat16
)
