#include "../normal.h"
#include "../random.h"
#include "../../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const NSUInteger metalRandomThreadsPerThreadgroup = 256;

int metal_dispatch_random_normal(
    MetalDeviceRef contextRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t seedLo,
    uint32_t seedHi,
    uint32_t ctrLo,
    uint32_t ctrHi,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_random_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (outRef == NULL) {
            metal_random_status_set(status, -2, "nil Metal output buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || context->device == NULL) {
            metal_random_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_random_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(
            context, "random_normal_float32", status
        );

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_random_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:0];
        [encoder setBytes:&count length:sizeof(count) atIndex:1];
        [encoder setBytes:&seedLo length:sizeof(seedLo) atIndex:2];
        [encoder setBytes:&seedHi length:sizeof(seedHi) atIndex:3];
        [encoder setBytes:&ctrLo length:sizeof(ctrLo) atIndex:4];
        [encoder setBytes:&ctrHi length:sizeof(ctrHi) atIndex:5];

        // Each thread emits 4 gaussians; total thread count is ceil(count / 4).
        NSUInteger threadCount = (NSUInteger)((count + 3u) / 4u);
        NSUInteger threadsPerGroup = metalRandomThreadsPerThreadgroup;
        if (threadCount < threadsPerGroup) {
            threadsPerGroup = threadCount;
        }
        NSUInteger threadgroupCount = (threadCount + threadsPerGroup - 1) / threadsPerGroup;

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
