#include "hawkes.h"
#include "../internal/bridge/core_private.h"


#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static const NSUInteger metalHMThreadCount = 256;

static void metal_hm_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_hm_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

static const char* metal_hm_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_hm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_hm_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_hm_status_set(status, -6, "unknown Metal Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_hm_status_set(status, -6, "Metal Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_hm_phase_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_hm_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        metal_hm_status_set(status, -6, "unknown Metal Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_hm_status_set(status, -6, "Metal Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_hm_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_hm_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_hm_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_hm_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_hm_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}


static NSUInteger metal_hm_thread_width(id<MTLComputePipelineState> pipeline) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];

    if (threadWidth == 0) {
        return metalHMThreadCount;
    }

    return threadWidth;
}

static int metal_hm_simple_prepare(
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

int metal_hm_pipeline(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->device == NULL) {
        metal_hm_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_hm_encode_hawkes_log_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef eventsRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef scratchRef,
    uint32_t eventCount,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)totalTimeRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:5];
    [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:6];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_hm_encode_hawkes_log_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)totalTimeRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
    [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_hm_encode_mi_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef jointRef,
    MetalBufferRef scratchRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)jointRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_hm_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}
