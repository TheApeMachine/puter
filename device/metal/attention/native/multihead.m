#include "multihead.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

static const char* metal_attention_variant_name(int variant) {
    switch (variant) {
    case 0: return "multi_head_attention";
    case 1: return "grouped_query_attention";
    case 2: return "sliding_window_attention";
    default: return NULL;
    }
}

int metal_dispatch_multi_head_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    int variant,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t numHeads,
    uint32_t kvHeads,
    uint32_t headDim,
    uint32_t windowSize,
    uint32_t causal,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_attention_status_clear(status);

        if (queryRef == NULL || keyRef == NULL || valueRef == NULL || outRef == NULL) {
            metal_transformer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        const char* operationName = metal_attention_variant_name(variant);

        if (operationName == NULL) {
            metal_transformer_status_set(status, -6, "unknown Metal attention variant");
            return -6;
        }

        MetalContext* context = (MetalContext*)contextRef;
        id<MTLComputePipelineState> pipeline = nil;
        int pipelineCode = metal_attention_pipeline(
            context, operationName, elementDType, status, &pipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder(context, &commandBuffer);

        if (encoder == nil) {
            return status != NULL && status->code != 0 ? status->code : -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:4];
        [encoder setBytes:&seqK length:sizeof(seqK) atIndex:5];
        [encoder setBytes:&numHeads length:sizeof(numHeads) atIndex:6];
        [encoder setBytes:&kvHeads length:sizeof(kvHeads) atIndex:7];
        [encoder setBytes:&headDim length:sizeof(headDim) atIndex:8];
        [encoder setBytes:&windowSize length:sizeof(windowSize) atIndex:9];
        [encoder setBytes:&causal length:sizeof(causal) atIndex:10];
        NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threads = 256;
        if (threads > maxThreads) {
            threads = maxThreads;
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(seqQ, numHeads, (headDim + 63) / 64)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);
        return 0;
    }
}
