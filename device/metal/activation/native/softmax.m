#include "softmax.h"
#include "activation.h"

#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static int metal_softmax_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal softmax kernel");
        return -6;
    }

    return metal_activation_compose_kernel_name(out, outBytes, "softmax", suffix, status);
}

int metal_dispatch_softmax(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_activation_status_clear(status);

        if (inputRef == NULL || outRef == NULL) {
            metal_activation_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_activation_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        char kernelName[128];
        int nameCode = metal_softmax_kernel_name(
            kernelName,
            sizeof(kernelName),
            elementDType,
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:2];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
