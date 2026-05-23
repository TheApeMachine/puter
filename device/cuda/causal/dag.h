#ifndef PUTER_DEVICE_CUDA_CAUSAL_DAG_H
#define PUTER_DEVICE_CUDA_CAUSAL_DAG_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_dag_markov_factorization(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef conditionalsRef,
    CUDABufferRef parentsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
