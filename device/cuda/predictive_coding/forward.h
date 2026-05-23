#ifndef PUTER_DEVICE_CUDA_PREDICTIVE_CODING_FORWARD_H
#define PUTER_DEVICE_CUDA_PREDICTIVE_CODING_FORWARD_H

#include "predictive_coding.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_pc_prediction(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_pc_prediction_error(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef observedRef,
    CUDABufferRef predictedRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
