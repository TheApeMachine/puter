#ifndef PUTER_DEVICE_CUDA_HAWKES_DISPATCH_H
#define PUTER_DEVICE_CUDA_HAWKES_DISPATCH_H

#include "hawkes.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_hawkes_register_module_source(const char* source);

const char* cuda_hawkes_module_source(void);

int cuda_hm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

int cuda_hm_phase_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    CUDAStatus* status
);

int cuda_dispatch_hawkes_intensity(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef eventsRef,
    CUDABufferRef queryTimesRef,
    CUDABufferRef baselineRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    uint32_t queryCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_hawkes_kernel_matrix(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef eventsRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_hawkes_log_likelihood(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef eventsRef,
    CUDABufferRef totalTimeRef,
    CUDABufferRef baselineRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_markov_blanket_partition(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef adjacencyRef,
    CUDABufferRef internalRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    uint32_t internalCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_markov_flow(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef mutualInformationRef,
    CUDABufferRef partitionRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    int32_t targetLabel,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_markov_mutual_information(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef jointRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
