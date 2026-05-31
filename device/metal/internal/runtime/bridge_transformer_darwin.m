#include "../bridge/bridge_transformer_private.h"
#include "../bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static void metal_transformer_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_transformer_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_transformer_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_transformer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_transformer_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_transformer_status_set(status, -6, "unknown Metal transformer kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_transformer_status_set(status, -6, "Metal transformer kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_transformer_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_transformer_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *pipeline = metal_get_pipeline(*context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static id<MTLBuffer> metal_transformer_validation_buffer(
    MetalContext* context,
    MetalStatus* status
) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLBuffer> validationBuffer = [device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared
    ];

    if (validationBuffer == nil) {
        metal_transformer_status_set(status, -9, "validation buffer allocation failed");
        return nil;
    }

    uint32_t zero = 0;
    memcpy([validationBuffer contents], &zero, sizeof(zero));
    return validationBuffer;
}

int metal_transformer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    bool needsValidation,
    uint64_t completionToken,
    MetalStatus* status,
    MetalTransformerEncodeBlock encode
) {
    @autoreleasepool {
        metal_transformer_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        MetalContext* context = NULL;
        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_transformer_prepare(
            contextRef, kernelName, status, &context, &queue, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLBuffer> validationBuffer = nil;
        if (needsValidation) {
            validationBuffer = metal_transformer_validation_buffer(context, status);

            if (validationBuffer == nil) {
                return -9;
            }
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        encode(encoder, validationBuffer);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion(
            (MetalContext*)contextRef,
            commandBuffer,
            completionToken,
            (__bridge void*)validationBuffer
        );
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
