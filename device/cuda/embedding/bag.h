#ifndef PUTER_DEVICE_CUDA_EMBEDDING_BAG_H
#define PUTER_DEVICE_CUDA_EMBEDDING_BAG_H

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_embedding_bag(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef tableRef,
    CUDABufferRef indicesRef,
    CUDABufferRef offsetsRef,
    CUDABufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
