#include "likelihood.h"
#include "hawkes.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_hawkes_log_likelihood(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (eventsRef == NULL || totalTimeRef == NULL || baselineRef == NULL ||
            alphaRef == NULL || betaRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_hm_phase_kernel_name(
            partialName, sizeof(partialName), "hawkes_log_likelihood", "partial", elementDType, status
        );
        int finalizeNameCode = metal_hm_phase_kernel_name(
            finalizeName, sizeof(finalizeName), "hawkes_log_likelihood", "finalize", elementDType, status
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
        int finalizePipelineCode = metal_hm_pipeline(contextRef, finalizeName, status, &finalizePipeline);
        if (finalizePipelineCode != 0) {
            return finalizePipelineCode;
        }

        int partialCode = metal_hm_encode_hawkes_log_partial(
            commandBuffer, partialPipeline, eventsRef, totalTimeRef, baselineRef, alphaRef, betaRef,
            scratchRef, eventCount, partialCount, status
        );
        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_hm_encode_hawkes_log_finalize(
            commandBuffer, finalizePipeline, scratchRef, totalTimeRef, baselineRef, outRef, eventCount, status
        );
        if (finalizeCode != 0) {
            return finalizeCode;
        }

        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
