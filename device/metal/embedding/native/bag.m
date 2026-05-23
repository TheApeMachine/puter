#include "bag.h"
#include "embedding.h"
#include "../internal/bridge/core_private.h"

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
) {
    if (tableRef == NULL || indicesRef == NULL || offsetsRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "embedding_bag", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)bagCount * hidden, true, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)tableRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)offsetsRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBuffer:validationBuffer offset:0 atIndex:4];
            [encoder setBytes:&vocab length:sizeof(vocab) atIndex:5];
            [encoder setBytes:&hidden length:sizeof(hidden) atIndex:6];
            [encoder setBytes:&indexCount length:sizeof(indexCount) atIndex:7];
            [encoder setBytes:&bagCount length:sizeof(bagCount) atIndex:8];
        }
    );
}
