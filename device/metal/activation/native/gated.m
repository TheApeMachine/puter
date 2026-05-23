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
            metal_activation_status_set(status, -6, "empty Metal gated dispatch");
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

static const char* metal_swiglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "swiglu_float32";
    case MetalElementDTypeFloat16:
        return "swiglu_float16";
    case MetalElementDTypeBFloat16:
        return "swiglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_swiglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_swiglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal swiglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_swiglu_packed_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "swiglu_packed_float32";
    case MetalElementDTypeFloat16:
        return "swiglu_packed_float16";
    case MetalElementDTypeBFloat16:
        return "swiglu_packed_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_swiglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_swiglu_packed_kernel_name(elementDType);

    if (destinationRef == NULL || packedRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal packed swiglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_geglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "geglu_float32";
    case MetalElementDTypeFloat16:
        return "geglu_float16";
    case MetalElementDTypeBFloat16:
        return "geglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_geglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_geglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal geglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_glu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "glu_float32";
    case MetalElementDTypeFloat16:
        return "glu_float16";
    case MetalElementDTypeBFloat16:
        return "glu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_glu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_glu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal glu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_reglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "reglu_float32";
    case MetalElementDTypeFloat16:
        return "reglu_float16";
    case MetalElementDTypeBFloat16:
        return "reglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_reglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_reglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal reglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_siglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "siglu_float32";
    case MetalElementDTypeFloat16:
        return "siglu_float16";
    case MetalElementDTypeBFloat16:
        return "siglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_siglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_siglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal siglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_seglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "seglu_float32";
    case MetalElementDTypeFloat16:
        return "seglu_float16";
    case MetalElementDTypeBFloat16:
        return "seglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_seglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_seglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal seglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_linglu_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "linglu_float32";
    case MetalElementDTypeFloat16:
        return "linglu_float16";
    case MetalElementDTypeBFloat16:
        return "linglu_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_linglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_linglu_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal linglu dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

static const char* metal_geglu_tanh_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "geglu_tanh_float32";
    case MetalElementDTypeFloat16:
        return "geglu_tanh_float16";
    case MetalElementDTypeBFloat16:
        return "geglu_tanh_bfloat16";
    default:
        return NULL;
    }
}

int metal_dispatch_geglu_tanh(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    const char* kernelName = metal_geglu_tanh_kernel_name(elementDType);

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL) {
        metal_activation_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    if (kernelName == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal geglu_tanh dtype");
        return -6;
    }

    return metal_gated_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
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

