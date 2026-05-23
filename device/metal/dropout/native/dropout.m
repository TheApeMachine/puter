#include "dropout.h"

#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

void metal_dropout_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_dropout_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_dropout_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_dropout_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_dropout_dtype_suffix(elementDType);
    if (suffix == NULL) {
        metal_dropout_status_set(status, -6, "unknown Metal dropout kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "dropout_%s", suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_dropout_status_set(status, -6, "Metal dropout kernel name overflow");
        return -6;
    }

    return 0;
}


static int metal_dropout_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_dropout_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_dropout_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

