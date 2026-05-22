#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_norm_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_norm_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_norm_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_norm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_norm_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_norm_status_set(status, -6, "unknown Metal normalization kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_norm_status_set(status, -6, "Metal normalization kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_norm_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal normalization command buffer failed";

        if (error != nil) {
            message = [NSString
                stringWithFormat:@"%@: %@",
                message,
                [error localizedDescription]
            ];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_norm_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    uint32_t rows,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_norm_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_norm_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        encode(encoder);
        [encoder
            dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_layernorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || scaleRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "layernorm",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        rows,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        }
    );
}

int metal_dispatch_rmsnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    float epsilon,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || scaleRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "rmsnorm",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        rows,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        }
    );
}

int metal_dispatch_adaptive_rmsnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef modulationRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || modulationRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "adaptive_rmsnorm",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        rows,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)modulationRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        }
    );
}

int metal_dispatch_modulated_layernorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef modulationRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t rowsPerBatch,
    uint32_t modulationCols,
    uint32_t modulationSet,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || modulationRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "modulated_layernorm",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        rows,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)modulationRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
            [encoder setBytes:&rowsPerBatch length:sizeof(rowsPerBatch) atIndex:4];
            [encoder setBytes:&modulationCols length:sizeof(modulationCols) atIndex:5];
            [encoder setBytes:&modulationSet length:sizeof(modulationSet) atIndex:6];
        }
    );
}

int metal_dispatch_gated_residual(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef residualRef,
    MetalBufferRef branchRef,
    MetalBufferRef modulationRef,
    MetalBufferRef outRef,
    uint32_t total,
    uint32_t cols,
    uint32_t rowsPerBatch,
    uint32_t modulationCols,
    uint32_t modulationSet,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (residualRef == NULL || branchRef == NULL || modulationRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "gated_residual",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        total / cols,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)residualRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)branchRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)modulationRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
            [encoder setBytes:&rowsPerBatch length:sizeof(rowsPerBatch) atIndex:5];
            [encoder setBytes:&modulationCols length:sizeof(modulationCols) atIndex:6];
            [encoder setBytes:&modulationSet length:sizeof(modulationSet) atIndex:7];
        }
    );
}

int metal_dispatch_batchnorm_denorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef meanRef,
    MetalBufferRef varianceRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || meanRef == NULL || varianceRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "batchnorm_denorm",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        rows,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)meanRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)varianceRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:4];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:5];
        }
    );
}

int metal_dispatch_groupnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || scaleRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName, sizeof(kernelName), "groupnorm", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        batch * groups,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:4];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:5];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:6];
        }
    );
}

int metal_dispatch_instancenorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || scaleRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName, sizeof(kernelName), "instancenorm", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        batch * channels,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:4];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:5];
        }
    );
}

int metal_dispatch_batchnorm_eval(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef meanRef,
    MetalBufferRef varianceRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || scaleRef == NULL || biasRef == NULL ||
        meanRef == NULL || varianceRef == NULL || outRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_norm_kernel_name(
        kernelName, sizeof(kernelName), "batchnorm_eval", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_norm_dispatch(
        contextRef,
        kernelName,
        batch * channels,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)meanRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)varianceRef offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:5];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:6];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:7];
        }
    );
}

int metal_dispatch_groupnorm_stats_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef meanRef,
    MetalBufferRef varianceRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || meanRef == NULL || varianceRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_norm_dispatch(
        contextRef,
        "groupnorm_stats_float32",
        batch * groups,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)meanRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)varianceRef offset:0 atIndex:2];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:3];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:4];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:5];
        }
    );
}

int metal_dispatch_instancenorm_stats_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef meanRef,
    MetalBufferRef varianceRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || meanRef == NULL || varianceRef == NULL) {
        metal_norm_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_norm_dispatch(
        contextRef,
        "instancenorm_stats_float32",
        batch * channels,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)meanRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)varianceRef offset:0 atIndex:2];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:3];
            [encoder setBytes:&spatial length:sizeof(spatial) atIndex:4];
        }
    );
}
