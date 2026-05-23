#include "int8.h"
#include "quant.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_quantization(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_quantization_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_quantization_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || context->device == NULL) {
            metal_quantization_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        const char* kernelName = metal_quantization_kernel_name(operation);

        if (kernelName == NULL) {
            metal_quantization_status_set(status, -6, "unknown Metal quantization kernel");
            return -6;
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
        [encoder setBytes:&count length:sizeof(count) atIndex:2];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
