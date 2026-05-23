#include "flash.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_flash_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_attention_status_clear(status);

        if (queryRef == NULL || keyRef == NULL || valueRef == NULL || outRef == NULL) {
            metal_transformer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;
        id<MTLComputePipelineState> pipeline = nil;
        int pipelineCode = metal_attention_pipeline(
            context, "flash_attention", elementDType, status, &pipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        id<MTLCommandBuffer> commandBuffer = metal_attention_command_buffer(context, status);

        if (commandBuffer == nil) {
            return status != NULL && status->code != 0 ? status->code : -3;
        }

        id<MTLComputeCommandEncoder> encoder =
            metal_attention_encoder(commandBuffer, pipeline, status);

        if (encoder == nil) {
            return status != NULL && status->code != 0 ? status->code : -4;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:4];
        [encoder setBytes:&seqK length:sizeof(seqK) atIndex:5];
        [encoder setBytes:&depth length:sizeof(depth) atIndex:6];
        [encoder setBytes:&valueDim length:sizeof(valueDim) atIndex:7];
        [encoder
            dispatchThreadgroups:MTLSizeMake(seqQ, (valueDim + 63) / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];
        return 0;
    }
}
