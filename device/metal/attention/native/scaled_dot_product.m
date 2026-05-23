#include "scaled_dot_product.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static int metal_attention_softmax_pipeline(
    MetalContext* context,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, "attention_softmax", status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_attention_prepare(
    MetalContext* context,
    int elementDType,
    MetalStatus* status,
    id<MTLComputePipelineState>* scoresPipeline,
    id<MTLComputePipelineState>* softmaxPipeline,
    id<MTLComputePipelineState>* weightedPipeline
) {
    int scoresCode = metal_attention_pipeline(
        context, "attention_scores", elementDType, status, scoresPipeline
    );

    if (scoresCode != 0) {
        return scoresCode;
    }

    int softmaxCode = metal_attention_softmax_pipeline(context, status, softmaxPipeline);

    if (softmaxCode != 0) {
        return softmaxCode;
    }

    return metal_attention_pipeline(
        context, "attention_weighted", elementDType, status, weightedPipeline
    );
}

static int metal_attention_encode_scores(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:2];
    [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:3];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:4];
    [encoder setBytes:&depth length:sizeof(depth) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake((seqK + 15) / 16, (seqQ + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
    ];
    [encoder endEncoding];
    return 0;
}

static int metal_attention_encode_softmax(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:1];
    [encoder
        dispatchThreadgroups:MTLSizeMake(seqQ, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
    ];
    [encoder endEncoding];
    return 0;
}

static int metal_attention_encode_weighted(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef valueRef,
    MetalBufferRef scoresRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t valueDim
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:3];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:4];
    [encoder setBytes:&valueDim length:sizeof(valueDim) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake((valueDim + 15) / 16, (seqQ + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
    ];
    [encoder endEncoding];
    return 0;
}

int metal_dispatch_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef scoresRef,
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

        if (queryRef == NULL || keyRef == NULL || valueRef == NULL ||
            scoresRef == NULL || outRef == NULL) {
            metal_transformer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;
        id<MTLComputePipelineState> scoresPipeline = nil;
        id<MTLComputePipelineState> softmaxPipeline = nil;
        id<MTLComputePipelineState> weightedPipeline = nil;
        int prepareCode = metal_attention_prepare(
            context,
            elementDType,
            status,
            &scoresPipeline,
            &softmaxPipeline,
            &weightedPipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = metal_attention_command_buffer(context, status);

        if (commandBuffer == nil) {
            return status != NULL && status->code != 0 ? status->code : -3;
        }

        int scoresCode = metal_attention_encode_scores(
            commandBuffer, scoresPipeline, status,
            queryRef, keyRef, scoresRef,
            seqQ, seqK, depth
        );

        if (scoresCode != 0) {
            return scoresCode;
        }

        int softmaxCode = metal_attention_encode_softmax(
            commandBuffer, softmaxPipeline, status, scoresRef, seqQ, seqK
        );

        if (softmaxCode != 0) {
            return softmaxCode;
        }

        int weightedCode = metal_attention_encode_weighted(
            commandBuffer, weightedPipeline, status,
            valueRef, scoresRef, outRef,
            seqQ, seqK, valueDim
        );

        if (weightedCode != 0) {
            return weightedCode;
        }

        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];
        return 0;
    }
}
