#ifndef PUTER_DEVICE_CUDA_MATMUL_MATMUL_CUH
#define PUTER_DEVICE_CUDA_MATMUL_MATMUL_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int matmulTileSizeCUDA = 16u;
static constexpr unsigned int matmulSharedFloatCountCUDA = 256u;

#endif
