#include "attention.h"

#include "../internal/bridge/bridge_transformer_private.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

void metal_attention_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

int metal_attention_pipeline(
    MetalContext* context,
    const char* operationName,
    int elementDType,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

id<MTLCommandBuffer> metal_attention_command_buffer(
    MetalContext* context,
    MetalStatus* status
) {
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_transformer_status_set(status, -1, "invalid Metal attention context");
        return nil;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    if (commandBuffer == nil) {
        metal_transformer_status_set(status, -3, "commandBuffer returned nil");
        return nil;
    }

    return commandBuffer;
}

id<MTLComputeCommandEncoder> metal_attention_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_transformer_status_set(status, -4, "computeCommandEncoder returned nil");
        return nil;
    }

    [encoder setComputePipelineState:pipeline];
    return encoder;
}
