#ifndef PUTER_DEVICE_METAL_EMBEDDING_BAG_H
#define PUTER_DEVICE_METAL_EMBEDDING_BAG_H

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_embedding_bag(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef offsetsRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
