#include "mask.h"
#include "dropout.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_dropout(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seed0,
    uint32_t seed1,
    uint32_t seed2,
    uint32_t seed3,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_dropout_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_dropout_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_dropout_kernel_name(kernelName, sizeof(kernelName), elementDType, status);
        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_dropout_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            metal_dropout_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        uint32_t seed[4] = {seed0, seed1, seed2, seed3};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:3];
        [encoder setBytes:&threshold length:sizeof(threshold) atIndex:4];
        [encoder setBytes:&seed length:sizeof(seed) atIndex:5];
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
