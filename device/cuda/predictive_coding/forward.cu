#include "predictive_coding.cuh"

PC_PREDICTION_KERNEL(pc_prediction_float32, pc_load_f32, pc_store_f32, float)
PC_PREDICTION_ERROR_KERNEL(pc_prediction_error_float32, pc_load_f32, pc_store_f32, float)

PC_PREDICTION_KERNEL(pc_prediction_float16, pc_load_f16, pc_store_f16, __half)
PC_PREDICTION_ERROR_KERNEL(pc_prediction_error_float16, pc_load_f16, pc_store_f16, __half)

PC_PREDICTION_KERNEL(pc_prediction_bfloat16, pc_load_bf16, pc_store_bf16, __nv_bfloat16)
PC_PREDICTION_ERROR_KERNEL(pc_prediction_error_bfloat16, pc_load_bf16, pc_store_bf16, __nv_bfloat16)
