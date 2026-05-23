#include "arithmetic.h"
#include "elementwise.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* metal_binary_operation_name(int operation) {
    switch (operation) {
    case MetalBinaryFloat32Add: return "add";
    case MetalBinaryFloat32Sub: return "sub";
    case MetalBinaryFloat32Mul: return "mul";
    case MetalBinaryFloat32Div: return "div";
    case MetalBinaryFloat32Max: return "max";
    case MetalBinaryFloat32Min: return "min";
    case MetalBinaryFloat32Eq: return "eq";
    case MetalBinaryFloat32Ne: return "ne";
    case MetalBinaryFloat32Lt: return "lt";
    case MetalBinaryFloat32Le: return "le";
    case MetalBinaryFloat32Gt: return "gt";
    case MetalBinaryFloat32Ge: return "ge";
    case MetalBinaryFloat32Pow: return "pow";
    case MetalBinaryFloat32Atan2: return "atan2";
    case MetalBinaryFloat32Mod: return "mod";
    default: return NULL;
    }
}

static int metal_elementwise_prepare(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    if (context == NULL || context->queue == NULL) {
        metal_elementwise_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_dispatch_binary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_elementwise_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
            metal_elementwise_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_elementwise_compose_kernel_name(
            kernelName,
            sizeof(kernelName),
            metal_binary_operation_name(operation),
            metal_elementwise_element_dtype_suffix(elementDType),
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_elementwise_prepare(
            (MetalContext*)contextRef,
            kernelName,
            status,
            &queue,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder
            dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
