#include "hawkes.h"
#include "../internal/bridge/core_private.h"


#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static NSUInteger metal_markov_thread_width(id<MTLComputePipelineState> pipeline) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];

    if (threadWidth == 0) {
        return 256;
    }

    return threadWidth;
}

static int metal_markov_prepare(
    MetalDeviceRef contextRef,
    const char* operationName,
    int elementDType,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    char kernelName[128];
    int nameCode = metal_hm_kernel_name(kernelName, sizeof(kernelName), operationName, elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_hm_prepare(contextRef, kernelName, status, commandBuffer, pipeline);
}

int metal_dispatch_markov_blanket_partition(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef internalRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t internalCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (adjacencyRef == NULL || internalRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_markov_prepare(
            contextRef, "markov_blanket_partition", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)adjacencyRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)internalRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&nodeCount length:sizeof(nodeCount) atIndex:3];
        [encoder setBytes:&internalCount length:sizeof(internalCount) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)nodeCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_markov_thread_width(pipeline), 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_markov_flow(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mutualInformationRef,
    MetalBufferRef partitionRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    int32_t targetLabel,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (mutualInformationRef == NULL || partitionRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_markov_prepare(
            contextRef, "markov_flow", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)mutualInformationRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)partitionRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&nodeCount length:sizeof(nodeCount) atIndex:3];
        [encoder setBytes:&targetLabel length:sizeof(targetLabel) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)nodeCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_markov_thread_width(pipeline), 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_markov_mutual_information(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef jointRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (jointRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_hm_phase_kernel_name(
            partialName, sizeof(partialName), "markov_mutual_information", "partial", elementDType, status
        );
        int finalizeNameCode = metal_hm_kernel_name(
            finalizeName, sizeof(finalizeName), "hawkes_markov_finalize", elementDType, status
        );

        if (partialNameCode != 0 || finalizeNameCode != 0) {
            return partialNameCode != 0 ? partialNameCode : finalizeNameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> partialPipeline = nil;
        int prepareCode = metal_hm_prepare(contextRef, partialName, status, &commandBuffer, &partialPipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int pipelineCode = metal_hm_pipeline(contextRef, finalizeName, status, &finalizePipeline);
        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int partialCode = metal_hm_encode_mi_partial(
            commandBuffer, partialPipeline, jointRef, scratchRef, rows, cols, partialCount, status
        );
        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_hm_encode_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, partialCount, status
        );
        if (finalizeCode != 0) {
            return finalizeCode;
        }

        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
