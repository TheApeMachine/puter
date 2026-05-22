#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_PRIVATE_H

#include "bridge_darwin.h"

#include <Metal/Metal.h>

typedef struct MetalDeferredCompletion {
    uint64_t token;
    void* validationBuffer;
} MetalDeferredCompletion;

typedef struct MetalContext {
    void* device;
    void* queue;
    void* library;
    void* pipelineCache;
    void* pipelineLock;
    bool isBatching;
    void* currentCommandBuffer;
    void* currentEncoder;
    int lastBatchStatus;
    MetalDeferredCompletion* deferredCompletions;
    size_t deferredCount;
    size_t deferredCapacity;
} MetalContext;

id<MTLComputeCommandEncoder> metal_get_encoder(MetalContext* context, id<MTLCommandBuffer>* outCommandBuffer);
void metal_suspend_compute_encoder(MetalContext* context);
void metal_end_encoder(MetalContext* context, id<MTLComputeCommandEncoder> encoder, id<MTLCommandBuffer> commandBuffer);
void metal_track_command_completion(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    uint64_t completionToken,
    void* validationBufferRef
);

id<MTLComputePipelineState> metal_get_pipeline(
    MetalContext* context,
    const char* name,
    MetalStatus* status
);

#endif
