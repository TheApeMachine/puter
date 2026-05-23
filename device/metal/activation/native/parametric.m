#include "parametric.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

int metal_dispatch_unary_param(
    MetalDeviceRef contextRef,
    const char* operationPrefix,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float param,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_activation_status_clear(status);

        if (count == 0 || operationPrefix == NULL) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || inputRef == NULL || outRef == NULL) {
            metal_activation_status_set(status, -2, "invalid Metal unary param dispatch");
            return -2;
        }

        const char* dtypeSuffix = metal_activation_element_dtype_suffix(elementDType);

        if (dtypeSuffix == NULL) {
            metal_activation_status_set(status, -6, "unknown Metal parametric dtype");
            return -6;
        }

        char kernelName[128];
        int nameCode = metal_activation_compose_kernel_name(
            kernelName,
            sizeof(kernelName),
            operationPrefix,
            dtypeSuffix,
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&param length:sizeof(param) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)metal_activation_vector_launch_count(count, elementDType);
        [encoder dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
