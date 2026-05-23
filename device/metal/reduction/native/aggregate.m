#include "aggregate.h"
#include "reduction.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const NSUInteger metalReductionThreadCountObjC = 256;

static int metal_reduction_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_reduction_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_reduction_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_reduction_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_reduction_encode_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    uint32_t count,
    uint32_t partialCount,
    uint32_t operation,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_reduction_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchARef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchBRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&operation length:sizeof(operation) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalReductionThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_reduction_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint32_t operation,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_reduction_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchARef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchBRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder setBytes:&operation length:sizeof(operation) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalReductionThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_reduction(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_reduction_status_clear(status);

        if (count == 0 || partialCount == 0) {
            return 0;
        }

        if (inputRef == NULL || scratchARef == NULL || scratchBRef == NULL || outRef == NULL) {
            metal_reduction_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_reduction_kernel_name(
            partialName, sizeof(partialName), "partial", elementDType, status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_reduction_kernel_name(
            finalizeName, sizeof(finalizeName), "finalize", elementDType, status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_reduction_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        int partialPipelineCode = metal_reduction_pipeline(
            context, partialName, status, &partialPipeline
        );

        if (partialPipelineCode != 0) {
            return partialPipelineCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int finalizePipelineCode = metal_reduction_pipeline(
            context, finalizeName, status, &finalizePipeline
        );

        if (finalizePipelineCode != 0) {
            return finalizePipelineCode;
        }

        uint32_t operationCode = (uint32_t)operation;
        int partialCode = metal_reduction_encode_partial(
            commandBuffer,
            partialPipeline,
            inputRef,
            scratchARef,
            scratchBRef,
            count,
            partialCount,
            operationCode,
            status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_reduction_encode_finalize(
            commandBuffer,
            finalizePipeline,
            scratchARef,
            scratchBRef,
            outRef,
            count,
            partialCount,
            operationCode,
            status
        );

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
