#include "bridge_transformer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_rope(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    float theta,
    float ropeFactor,
    float lowFreqFactor,
    float highFreqFactor,
    uint32_t originalContext,
    uint32_t halfMode,
    uint32_t positionOffset,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "rope", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)pairCount, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:2];
            [encoder setBytes:&numHeads length:sizeof(numHeads) atIndex:3];
            [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
            [encoder setBytes:&pairCount length:sizeof(pairCount) atIndex:5];
            [encoder setBytes:&theta length:sizeof(theta) atIndex:6];
            [encoder setBytes:&ropeFactor length:sizeof(ropeFactor) atIndex:7];
            [encoder setBytes:&lowFreqFactor length:sizeof(lowFreqFactor) atIndex:8];
            [encoder setBytes:&highFreqFactor length:sizeof(highFreqFactor) atIndex:9];
            [encoder setBytes:&originalContext length:sizeof(originalContext) atIndex:10];
            [encoder setBytes:&halfMode length:sizeof(halfMode) atIndex:11];
            [encoder setBytes:&positionOffset length:sizeof(positionOffset) atIndex:12];
        }
    );
}

int metal_dispatch_multi_axis_rope(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    uint32_t latentSeqLen,
    uint32_t latentSide,
    float theta,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "multi_axis_rope", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)pairCount, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&seqLen length:sizeof(seqLen) atIndex:2];
            [encoder setBytes:&numHeads length:sizeof(numHeads) atIndex:3];
            [encoder setBytes:&headDim length:sizeof(headDim) atIndex:4];
            [encoder setBytes:&pairCount length:sizeof(pairCount) atIndex:5];
            [encoder setBytes:&latentSeqLen length:sizeof(latentSeqLen) atIndex:6];
            [encoder setBytes:&latentSide length:sizeof(latentSide) atIndex:7];
            [encoder setBytes:&theta length:sizeof(theta) atIndex:8];
        }
    );
}
