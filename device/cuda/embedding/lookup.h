#ifndef PUTER_DEVICE_CUDA_EMBEDDING_LOOKUP_H
#define PUTER_DEVICE_CUDA_EMBEDDING_LOOKUP_H

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_embedding_lookup(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef tableRef,
    CUDABufferRef indicesRef,
    CUDABufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
