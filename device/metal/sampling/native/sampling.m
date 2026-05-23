#include "sampling.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static const NSUInteger metalSamplingThreadCountObjC = 256;

void metal_sampling_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_sampling_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_sampling_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_sampling_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_sampling_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_sampling_status_set(status, -6, "unknown Metal sampling kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_sampling_status_set(status, -6, "Metal sampling kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_sampling_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_sampling_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_sampling_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_sampling_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_sampling_encode_greedy(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logitsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalSamplingThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_init(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t count,
    uint32_t paddedCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logitsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&paddedCount length:sizeof(paddedCount) atIndex:4];
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = 1;
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)paddedCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_bitonic(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t stageSize,
    uint32_t passSize,
    uint32_t paddedCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
    [encoder setBytes:&stageSize length:sizeof(stageSize) atIndex:2];
    [encoder setBytes:&passSize length:sizeof(passSize) atIndex:3];
    [encoder setBytes:&paddedCount length:sizeof(paddedCount) atIndex:4];
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = 1;
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)paddedCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_draw(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    float target,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&target length:sizeof(target) atIndex:4];
    [encoder
        dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_sort(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t paddedCount,
    MetalStatus* status
) {
    for (uint32_t stageSize = 2; stageSize <= paddedCount; stageSize <<= 1) {
        for (uint32_t passSize = stageSize >> 1; passSize > 0; passSize >>= 1) {
            int code = metal_sampling_encode_bitonic(
                commandBuffer, pipeline, scoresRef, indicesRef, stageSize, passSize, paddedCount, status
            );

            if (code != 0) {
                return code;
            }
        }

        if (stageSize == paddedCount) {
            break;
        }
    }

    return 0;
}

static int metal_sampling_dispatch_greedy(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
) {
    char kernelName[128];
    int nameCode = metal_sampling_kernel_name(
        kernelName, sizeof(kernelName), "greedy_sample", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    id<MTLComputePipelineState> pipeline = nil;
    int pipelineCode = metal_sampling_pipeline(context, kernelName, status, &pipeline);

    if (pipelineCode != 0) {
        return pipelineCode;
    }

    return metal_sampling_encode_greedy(commandBuffer, pipeline, logitsRef, outRef, count, status);
}

static int metal_sampling_dispatch_draw(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    MetalStatus* status
) {
    char initName[128];
    int nameCode = metal_sampling_kernel_name(
        initName, sizeof(initName), "sampling_init", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    id<MTLComputePipelineState> initPipeline = nil;
    int initCode = metal_sampling_pipeline(context, initName, status, &initPipeline);

    if (initCode != 0) {
        return initCode;
    }

    id<MTLComputePipelineState> bitonicPipeline = nil;
    int bitonicCode = metal_sampling_pipeline(context, "sampling_bitonic_step", status, &bitonicPipeline);

    if (bitonicCode != 0) {
        return bitonicCode;
    }

    id<MTLComputePipelineState> drawPipeline = nil;
    int drawCode = metal_sampling_pipeline(context, "sampling_draw_sorted", status, &drawPipeline);

    if (drawCode != 0) {
        return drawCode;
    }

    int encodeInitCode = metal_sampling_encode_init(
        commandBuffer, initPipeline, logitsRef, scoresRef, indicesRef, count, paddedCount, status
    );

    if (encodeInitCode != 0) {
        return encodeInitCode;
    }

    int sortCode = metal_sampling_encode_sort(
        commandBuffer, bitonicPipeline, scoresRef, indicesRef, paddedCount, status
    );

    if (sortCode != 0) {
        return sortCode;
    }

    return metal_sampling_encode_draw(
        commandBuffer, drawPipeline, scoresRef, indicesRef, outRef, count, target, status
    );
}

int metal_dispatch_sampling(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_sampling_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (logitsRef == NULL || outRef == NULL) {
            metal_sampling_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        if (operation != 0 && (scoresRef == NULL || indicesRef == NULL)) {
            metal_sampling_status_set(status, -2, "nil Metal sampling scratch buffer");
            return -2;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_sampling_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        int dispatchCode = 0;
        if (operation == 0) {
            dispatchCode = metal_sampling_dispatch_greedy(
                context, commandBuffer, elementDType, logitsRef, outRef, count, status
            );
        }

        if (operation != 0) {
            dispatchCode = metal_sampling_dispatch_draw(
                context,
                commandBuffer,
                elementDType,
                logitsRef,
                scoresRef,
                indicesRef,
                outRef,
                count,
                paddedCount,
                target,
                status
            );
        }

        if (dispatchCode != 0) {
            return dispatchCode;
        }

        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
