#include "intensity.h"
#include "hawkes.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_hawkes_intensity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef queryTimesRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t queryCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_hm_status_clear(status);

        if (eventsRef == NULL || queryTimesRef == NULL || baselineRef == NULL ||
            alphaRef == NULL || betaRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_hm_simple_prepare(
            contextRef, "hawkes_intensity", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)queryTimesRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:5];
        [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:6];
        [encoder
            dispatchThreadgroups:MTLSizeMake((NSUInteger)queryCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
