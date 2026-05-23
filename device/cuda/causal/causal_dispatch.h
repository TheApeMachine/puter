#ifndef PUTER_DEVICE_CUDA_CAUSAL_DISPATCH_H
#define PUTER_DEVICE_CUDA_CAUSAL_DISPATCH_H

#include "adjustment.h"
#include "dag.h"
#include "intervention.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_causal_register_module_source(const char* source);

const char* cuda_causal_module_source(void);

int cuda_causal_named_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    const char* operationName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_causal_two_phase_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    const char* operationName,
    uint32_t partialGridX,
    void** partialArgs,
    size_t partialArgsBytes,
    void** finalizeArgs,
    size_t finalizeArgsBytes,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_causal_dag_two_phase_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    uint32_t partialGridX,
    void** partialArgs,
    size_t partialArgsBytes,
    void** finalizeArgs,
    size_t finalizeArgsBytes,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
