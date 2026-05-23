#ifndef PUTER_DEVICE_CUDA_REDUCTION_REDUCTION_CUH
#define PUTER_DEVICE_CUDA_REDUCTION_REDUCTION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int reductionThreadCountCUDA = 256u;

#endif
