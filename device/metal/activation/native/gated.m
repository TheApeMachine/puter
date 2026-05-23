#include "gated.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static int metal_gated_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_activation_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        if (kernelName == NULL) {
            metal_activation_status_set(status, -6, "unknown Metal gated kernel");
            return -6;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || context->device == NULL) {
            metal_activation_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);
        [encoder setComputePipelineState:pipeline];
        encode(encoder);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);
        return 0;
    }
}

static int metal_gated_launch_tensor(
    MetalDeviceRef contextRef,
    const char* kernelName,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)metal_activation_vector_launch_count(count, elementDType),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)destinationRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gateRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)upRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}

static int metal_gated_launch_packed(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (destinationRef == NULL || packedRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)count,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)destinationRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)packedRef offset:0 atIndex:1];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}

static const char* metal_gated_tensor_kernel_name(const char* prefix, int elementDType, MetalStatus* status) {
    static __thread char kernelName[128];
    const char* suffix = metal_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal gated dtype");
        return NULL;
    }

    if (metal_activation_compose_kernel_name(kernelName, sizeof(kernelName), prefix, suffix, status) != 0) {
        return NULL;
    }

    return kernelName;
}

static const char* metal_gated_packed_kernel_name(const char* prefix, int elementDType, MetalStatus* status) {
    static __thread char kernelName[128];
    const char* suffix = metal_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal packed gated dtype");
        return NULL;
    }

    char prefixBuffer[64];
    snprintf(prefixBuffer, sizeof(prefixBuffer), "%s_packed", prefix);

    if (metal_activation_compose_kernel_name(kernelName, sizeof(kernelName), prefixBuffer, suffix, status) != 0) {
        return NULL;
    }

    return kernelName;
}

#define METAL_DISPATCH_GATED_TENSOR(name) \
int metal_dispatch_##name( \
    MetalDeviceRef contextRef, \
    int elementDType, \
    MetalBufferRef destinationRef, \
    MetalBufferRef gateRef, \
    MetalBufferRef upRef, \
    uint32_t count, \
    uint64_t completionToken, \
    MetalStatus* status \
) { \
    const char* kernelName = metal_gated_tensor_kernel_name(#name, elementDType, status); \
    if (kernelName == NULL) { \
        return status != NULL && status->code != 0 ? status->code : -6; \
    } \
    return metal_gated_launch_tensor( \
        contextRef, \
        kernelName, \
        elementDType, \
        destinationRef, \
        gateRef, \
        upRef, \
        count, \
        completionToken, \
        status \
    ); \
}

#define METAL_DISPATCH_GATED_PACKED(name) \
int metal_dispatch_##name##_packed( \
    MetalDeviceRef contextRef, \
    int elementDType, \
    MetalBufferRef destinationRef, \
    MetalBufferRef packedRef, \
    uint32_t inner, \
    uint32_t count, \
    uint64_t completionToken, \
    MetalStatus* status \
) { \
    const char* kernelName = metal_gated_packed_kernel_name(#name, elementDType, status); \
    if (kernelName == NULL) { \
        return status != NULL && status->code != 0 ? status->code : -6; \
    } \
    return metal_gated_launch_packed( \
        contextRef, \
        kernelName, \
        destinationRef, \
        packedRef, \
        inner, \
        count, \
        completionToken, \
        status \
    ); \
}

METAL_DISPATCH_GATED_TENSOR(swiglu)
METAL_DISPATCH_GATED_PACKED(swiglu)
METAL_DISPATCH_GATED_TENSOR(geglu)
METAL_DISPATCH_GATED_PACKED(geglu)
METAL_DISPATCH_GATED_TENSOR(glu)
METAL_DISPATCH_GATED_PACKED(glu)
METAL_DISPATCH_GATED_TENSOR(reglu)
METAL_DISPATCH_GATED_PACKED(reglu)
METAL_DISPATCH_GATED_TENSOR(siglu)
METAL_DISPATCH_GATED_PACKED(siglu)
METAL_DISPATCH_GATED_TENSOR(seglu)
METAL_DISPATCH_GATED_PACKED(seglu)
METAL_DISPATCH_GATED_TENSOR(linglu)
METAL_DISPATCH_GATED_PACKED(linglu)
METAL_DISPATCH_GATED_TENSOR(geglu_tanh)
METAL_DISPATCH_GATED_PACKED(geglu_tanh)

int metal_dispatch_timestep_embedding(
    MetalDeviceRef contextRef,
    MetalBufferRef timestepsRef,
    MetalBufferRef maxPeriodRef,
    MetalBufferRef downscaleRef,
    MetalBufferRef flipRef,
    MetalBufferRef outRef,
    int elementDType,
    uint32_t count,
    uint32_t dim,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (
        timestepsRef == NULL ||
        maxPeriodRef == NULL ||
        downscaleRef == NULL ||
        flipRef == NULL ||
        outRef == NULL
    ) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    const char* kernelName = NULL;

    switch (elementDType) {
    case MetalElementDTypeFloat32:
        kernelName = "timestep_embedding_float32";
        break;
    case MetalElementDTypeFloat16:
        kernelName = "timestep_embedding_float16";
        break;
    case MetalElementDTypeBFloat16:
        kernelName = "timestep_embedding_bfloat16";
        break;
    default:
        metal_activation_status_set(status, -6, "unknown Metal timestep dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)(count * dim),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)timestepsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)maxPeriodRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)downscaleRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)flipRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
            [encoder setBytes:&dim length:sizeof(dim) atIndex:6];
        }
    );
}
