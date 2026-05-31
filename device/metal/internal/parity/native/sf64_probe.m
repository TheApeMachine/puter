#include "../sf64_probe.h"
#include "../../bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <string.h>

static void sf64_probe_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void sf64_probe_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    strncpy(status->message, message, sizeof(status->message) - 1);
    status->message[sizeof(status->message) - 1] = '\0';
}

int metal_dispatch_sf64_transcendental_probe(
    MetalDeviceRef contextRef,
    MetalBufferRef inputsRef,
    MetalBufferRef sqrtInputsRef,
    MetalBufferRef outputsRef,
    uint32_t caseCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        sf64_probe_status_clear(status);

        if (caseCount == 0) {
            return 0;
        }

        if (inputsRef == NULL || sqrtInputsRef == NULL || outputsRef == NULL) {
            sf64_probe_status_set(status, -2, "nil Metal probe buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || context->device == NULL) {
            sf64_probe_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            sf64_probe_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(
            context, "sf64_transcendental_probe", status
        );

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            sf64_probe_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputsRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sqrtInputsRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outputsRef offset:0 atIndex:2];
        [encoder setBytes:&caseCount length:sizeof(caseCount) atIndex:3];

        NSUInteger threadsPerGroup = MIN((NSUInteger)caseCount, 256u);
        NSUInteger threadgroupCount = ((NSUInteger)caseCount + threadsPerGroup - 1) / threadsPerGroup;

        [encoder
            dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)
        ];
        [encoder endEncoding];

        metal_track_command_completion(context, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
