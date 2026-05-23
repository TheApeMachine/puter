#include "lookup.h"
#include "embedding.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_embedding_lookup(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (tableRef == NULL || indicesRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "embedding_lookup", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)indexCount * hidden, true, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)tableRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBuffer:validationBuffer offset:0 atIndex:3];
            [encoder setBytes:&vocab length:sizeof(vocab) atIndex:4];
            [encoder setBytes:&hidden length:sizeof(hidden) atIndex:5];
            [encoder setBytes:&indexCount length:sizeof(indexCount) atIndex:6];
        }
    );
}
