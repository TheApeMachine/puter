#include "kernel.h"
#include "hawkes.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_hawkes_kernel_matrix(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_hm_status_clear(status);

        if (eventsRef == NULL || alphaRef == NULL || betaRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_hm_simple_prepare(
            contextRef, "hawkes_kernel_matrix", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        uint32_t total = eventCount * eventCount;
        [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_hm_thread_width(pipeline), 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
