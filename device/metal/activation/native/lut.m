#include "lut.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const char* metal_lut_gather_kernel_name(int elementDType, MetalStatus* status) {
    switch (elementDType) {
    case MetalElementDTypeFloat16:
        return "lut_gather_float16";
    case MetalElementDTypeBFloat16:
        return "lut_gather_bfloat16";
    default:
        metal_activation_status_set(status, -6, "unsupported Metal LUT gather dtype");
        return NULL;
    }
}

int metal_dispatch_lut_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    MetalBufferRef lutRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_activation_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL || lutRef == NULL) {
            metal_activation_status_set(status, -2, "nil Metal LUT buffer");
            return -2;
        }

        const char* kernelName = metal_lut_gather_kernel_name(elementDType, status);

        if (kernelName == NULL) {
            return status != NULL && status->code != 0 ? status->code : -6;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_activation_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder(context, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)lutRef offset:0 atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger launchCount = (NSUInteger)((count + 7) / 8);
        [encoder
            dispatchThreads:MTLSizeMake(launchCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion(context, commandBuffer, completionToken, NULL);
        metal_end_encoder(context, encoder, commandBuffer);

        return 0;
    }
}
