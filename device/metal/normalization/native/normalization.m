#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
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

// metal_dispatch_rmsnorm previously lived here as a pre-quintet
// leftover (see GAPS.md §6.6.1: duplicate cgo symbols from before
// the family-quintet reorganization landed). RMSNorm belongs to the
// `layernorm` family per ARCHITECTURE.md §2.3 — the canonical
// implementation now lives in device/metal/layernorm/native/layer.m
// and is exposed to Go via layernorm.DispatchRMSNormRefs. Removed
// from here to clear the duplicate-symbol link error.

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

    uint32_t rowsPerBatch = rows;
    uint32_t modulationCols = cols * 2;
    float epsilon = 1.0e-6f;

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
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:6];
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
    float epsilon,
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
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:7];
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

static int metal_groupnorm_dispatch_f32(
    MetalDeviceRef contextRef,
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
    @autoreleasepool {
        metal_norm_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_norm_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        uint32_t rowCount = batch * groups;
        long long statsBytes = (long long)rowCount * 2LL * (long long)sizeof(float);
        MetalBufferRef statsRef = metal_buffer_new_shared(contextRef, statsBytes);

        if (statsRef == NULL) {
            metal_norm_status_set(status, -3, "groupnorm stats buffer allocation failed");
            return -3;
        }

        id<MTLComputePipelineState> statsPipeline =
            metal_get_pipeline(context, "groupnorm_stats_float32", status);
        id<MTLComputePipelineState> applyPipeline =
            metal_get_pipeline(context, "groupnorm_apply_float32", status);

        if (statsPipeline == nil || applyPipeline == nil) {
            metal_buffer_release(statsRef);
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:statsPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)statsRef offset:0 atIndex:1];
        [encoder setBytes:&channels length:sizeof(channels) atIndex:2];
        [encoder setBytes:&spatial length:sizeof(spatial) atIndex:3];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:4];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:applyPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)statsRef offset:0 atIndex:4];
        [encoder setBytes:&channels length:sizeof(channels) atIndex:5];
        [encoder setBytes:&spatial length:sizeof(spatial) atIndex:6];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:7];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];

        metal_track_command_completion(contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];
        metal_buffer_release(statsRef);

        return 0;
    }
}

static int metal_groupnorm_dispatch_f16(
    MetalDeviceRef contextRef,
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
    @autoreleasepool {
        metal_norm_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_norm_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        uint32_t rowCount = batch * groups;
        long long statsBytes = (long long)rowCount * 2LL * (long long)sizeof(float);
        MetalBufferRef statsRef = metal_buffer_new_shared(contextRef, statsBytes);

        if (statsRef == NULL) {
            metal_norm_status_set(status, -3, "groupnorm stats buffer allocation failed");
            return -3;
        }

        id<MTLComputePipelineState> statsPipeline =
            metal_get_pipeline(context, "groupnorm_stats_float16", status);
        id<MTLComputePipelineState> applyPipeline =
            metal_get_pipeline(context, "groupnorm_apply_float16", status);

        if (statsPipeline == nil || applyPipeline == nil) {
            metal_buffer_release(statsRef);
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:statsPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)statsRef offset:0 atIndex:1];
        [encoder setBytes:&channels length:sizeof(channels) atIndex:2];
        [encoder setBytes:&spatial length:sizeof(spatial) atIndex:3];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:4];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_buffer_release(statsRef);
            metal_norm_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:applyPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)statsRef offset:0 atIndex:4];
        [encoder setBytes:&channels length:sizeof(channels) atIndex:5];
        [encoder setBytes:&spatial length:sizeof(spatial) atIndex:6];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:7];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];

        metal_track_command_completion(contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];
        metal_buffer_release(statsRef);

        return 0;
    }
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

    if (elementDType == MetalElementDTypeFloat32) {
        return metal_groupnorm_dispatch_f32(
            contextRef,
            inputRef,
            scaleRef,
            biasRef,
            outRef,
            batch,
            channels,
            spatial,
            groups,
            completionToken,
            status
        );
    }

    if (elementDType == MetalElementDTypeFloat16) {
        return metal_groupnorm_dispatch_f16(
            contextRef,
            inputRef,
            scaleRef,
            biasRef,
            outRef,
            batch,
            channels,
            spatial,
            groups,
            completionToken,
            status
        );
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
