#include "adjustment.h"
#include "causal.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_backdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (conditionalRef == NULL || marginalRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "backdoor_adjustment", (NSUInteger)xCount * yCount,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)conditionalRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)marginalRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&xCount length:sizeof(xCount) atIndex:3];
            [encoder setBytes:&zCount length:sizeof(zCount) atIndex:4];
            [encoder setBytes:&yCount length:sizeof(yCount) atIndex:5];
        }
    );
}

int metal_dispatch_frontdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mediatorRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (mediatorRef == NULL || outcomeRef == NULL || marginalRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "frontdoor_adjustment", (NSUInteger)xCount * yCount,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)mediatorRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outcomeRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)marginalRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&xCount length:sizeof(xCount) atIndex:4];
            [encoder setBytes:&mCount length:sizeof(mCount) atIndex:5];
            [encoder setBytes:&yCount length:sizeof(yCount) atIndex:6];
        }
    );
}
