#include "convolution.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

typedef void (^MetalConvolutionEncodeBlock)(id<MTLComputeCommandEncoder> encoder);

static void metal_convolution_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_convolution_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_convolution_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_convolution_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_convolution_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_convolution_status_set(status, -6, "unknown Metal convolution kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_convolution_status_set(status, -6, "Metal convolution kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_convolution_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_convolution_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_convolution_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalConvolutionEncodeBlock encode
) {
    @autoreleasepool {
        metal_convolution_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_convolution_prepare(contextRef, kernelName, status, &queue, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        encode(encoder);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
#include "convolution.h"
#include "../internal/bridge/core_private.h"
