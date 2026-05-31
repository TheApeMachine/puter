#include "update.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static void metal_resonant_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_resonant_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_resonant_element_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "fp32";
    case MetalElementDTypeFloat16:
        return "fp16";
    case MetalElementDTypeBFloat16:
        return "bfloat16";
    default:
        return NULL;
    }
}

static int metal_resonant_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_resonant_element_dtype_suffix(elementDType);

    if (prefix == NULL || suffix == NULL) {
        metal_resonant_status_set(status, -6, "unsupported resonant dtype");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_resonant_status_set(status, -6, "resonant kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_resonant_prepare(
    MetalDeviceRef contextRef,
    const char* prefix,
    int elementDType,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    char kernelName[128];
    int nameCode = metal_resonant_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        prefix,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_resonant_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_resonant_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_resonant_validate_params(const MetalResonantUpdateParams* params, MetalStatus* status) {
    if (params == NULL || params->n == 0 || params->D == 0 || params->H == 0) {
        metal_resonant_status_set(status, -2, "invalid resonant update params");
        return -2;
    }

    return 0;
}

int metal_dispatch_resonant_update_forward(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef xRef,
    MetalBufferRef yRef,
    MetalBufferRef vrRef,
    MetalBufferRef viRef,
    MetalBufferRef diagRef,
    MetalBufferRef xOutRef,
    MetalBufferRef yOutRef,
    MetalBufferRef aOutRef,
    MetalBufferRef bOutRef,
    MetalBufferRef invROutRef,
    const MetalResonantUpdateParams* params,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_resonant_status_clear(status);

        int paramsCode = metal_resonant_validate_params(params, status);

        if (paramsCode != 0) {
            return paramsCode;
        }

        if (
            xRef == NULL || yRef == NULL || vrRef == NULL || viRef == NULL || diagRef == NULL
            || xOutRef == NULL || yOutRef == NULL || aOutRef == NULL || bOutRef == NULL || invROutRef == NULL
        ) {
            metal_resonant_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_resonant_prepare(
            contextRef,
            "resonant_update_fwd",
            elementDType,
            status,
            &commandBuffer,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_resonant_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        MetalResonantUpdateParams paramsCopy = *params;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)yRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)vrRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)viRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)diagRef offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)xOutRef offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)yOutRef offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)aOutRef offset:0 atIndex:7];
        [encoder setBuffer:(__bridge id<MTLBuffer>)bOutRef offset:0 atIndex:8];
        [encoder setBuffer:(__bridge id<MTLBuffer>)invROutRef offset:0 atIndex:9];
        [encoder setBytes:&paramsCopy length:sizeof(paramsCopy) atIndex:10];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)paramsCopy.n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_resonant_update_backward(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef gradXOutRef,
    MetalBufferRef gradYOutRef,
    MetalBufferRef xRef,
    MetalBufferRef yRef,
    MetalBufferRef diagRef,
    MetalBufferRef aRef,
    MetalBufferRef bRef,
    MetalBufferRef invRRef,
    MetalBufferRef gradXRef,
    MetalBufferRef gradYRef,
    MetalBufferRef gradVRRef,
    MetalBufferRef gradVIRef,
    const MetalResonantUpdateParams* params,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_resonant_status_clear(status);

        int paramsCode = metal_resonant_validate_params(params, status);

        if (paramsCode != 0) {
            return paramsCode;
        }

        if (
            gradXOutRef == NULL || gradYOutRef == NULL || xRef == NULL || yRef == NULL || diagRef == NULL
            || aRef == NULL || bRef == NULL || invRRef == NULL || gradXRef == NULL || gradYRef == NULL
            || gradVRRef == NULL || gradVIRef == NULL
        ) {
            metal_resonant_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_resonant_prepare(
            contextRef,
            "resonant_update_bwd",
            elementDType,
            status,
            &commandBuffer,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_resonant_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        MetalResonantUpdateParams paramsCopy = *params;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradXOutRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradYOutRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)yRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)diagRef offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)invRRef offset:0 atIndex:7];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradVRRef offset:0 atIndex:8];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradVIRef offset:0 atIndex:9];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradXRef offset:0 atIndex:10];
        [encoder setBuffer:(__bridge id<MTLBuffer>)gradYRef offset:0 atIndex:11];
        [encoder setBytes:&paramsCopy length:sizeof(paramsCopy) atIndex:12];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)paramsCopy.n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
