#ifndef PUTER_DEVICE_CUDA_PREDICTIVE_CODING_LEARNING_H
#define PUTER_DEVICE_CUDA_PREDICTIVE_CODING_LEARNING_H

#include "predictive_coding.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_pc_update_representation(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef errorRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    float learningRate,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_pc_update_weights(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef errorRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    float learningRate,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
