#include "bridge_optimizer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static void metal_optimizer_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_optimizer_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_optimizer_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "float32";
    case MetalElementDTypeFloat16:
        return "float16";
    case MetalElementDTypeBFloat16:
        return "bfloat16";
    default:
        return NULL;
    }
}

int metal_optimizer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_optimizer_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_optimizer_status_set(status, -6, "unknown Metal optimizer kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_optimizer_status_set(status, -6, "Metal optimizer kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_optimizer_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_optimizer_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_optimizer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalOptimizerEncodeBlock encode
) {
    @autoreleasepool {
        metal_optimizer_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_optimizer_prepare(
            contextRef, kernelName, status, &queue, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        if (encoder == nil || commandBuffer == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

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
        metal_track_command_completion(
            (MetalContext*)contextRef,
            commandBuffer,
            completionToken,
            NULL
        );
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

static const char* metal_optimizer4_operation_name(int operation) {
    switch (operation) {
    case 0:
        return "adam_step";
    case 1:
        return "adamw_step";
    case 2:
        return "adamax_step";
    default:
        return NULL;
    }
}

int metal_dispatch_optimizer4(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef firstRef,
    MetalBufferRef secondRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (paramsRef == NULL || gradientsRef == NULL || firstRef == NULL ||
        secondRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    const char* operationName = metal_optimizer4_operation_name(operation);

    if (operationName == NULL) {
        metal_optimizer_status_set(status, -5, "unknown optimizer4 operation");
        return -5;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)firstRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)secondRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
            [encoder setBytes:configBytes length:configBytesLen atIndex:6];
        }
    );
}

int metal_dispatch_optimizer3(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (paramsRef == NULL || gradientsRef == NULL || stateRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName, sizeof(kernelName), "optimizer3", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    uint32_t operationValue = (uint32_t)operation;

    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)stateRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
            [encoder setBytes:&operationValue length:sizeof(operationValue) atIndex:5];
            [encoder setBytes:configBytes length:configBytesLen atIndex:6];
        }
    );
}

int metal_dispatch_optimizer2(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
) {
    (void)operation;

    if (paramsRef == NULL || gradientsRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName, sizeof(kernelName), "lbfgs_step", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:configBytes length:configBytesLen atIndex:4];
        }
    );
}

int metal_dispatch_hebbian_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef postRef,
    MetalBufferRef preRef,
    MetalBufferRef outRef,
    uint32_t postCount,
    uint32_t preCount,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (weightsRef == NULL || postRef == NULL || preRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName, sizeof(kernelName), "hebbian_step", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)postCount * preCount;
    return metal_optimizer_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)postRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)preRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&postCount length:sizeof(postCount) atIndex:4];
            [encoder setBytes:&preCount length:sizeof(preCount) atIndex:5];
            [encoder setBytes:configBytes length:configBytesLen atIndex:6];
        }
    );
}

int metal_dispatch_lars_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef momentumRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t groupCount,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (paramsRef == NULL || gradientsRef == NULL || momentumRef == NULL ||
            scratchRef == NULL || outRef == NULL) {
            metal_optimizer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char normsName[128];
        int normsNameCode = metal_optimizer_kernel_name(
            normsName, sizeof(normsName), "lars_norms", elementDType, status
        );

        if (normsNameCode != 0) {
            return normsNameCode;
        }

        char stepName[128];
        int stepNameCode = metal_optimizer_kernel_name(
            stepName, sizeof(stepName), "lars_step", elementDType, status
        );

        if (stepNameCode != 0) {
            return stepNameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> normsPipeline = nil;
        int normsPrepare = metal_optimizer_prepare(
            contextRef, normsName, status, &queue, &normsPipeline
        );

        if (normsPrepare != 0) {
            return normsPrepare;
        }

        id<MTLCommandQueue> stepQueue = nil;
        id<MTLComputePipelineState> stepPipeline = nil;
        int stepPrepare = metal_optimizer_prepare(
            contextRef, stepName, status, &stepQueue, &stepPipeline
        );

        if (stepPrepare != 0) {
            return stepPrepare;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_optimizer_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> normsEncoder = [commandBuffer computeCommandEncoder];

        if (normsEncoder == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [normsEncoder setComputePipelineState:normsPipeline];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
        [normsEncoder setBytes:&count length:sizeof(count) atIndex:3];
        [normsEncoder
            dispatchThreadgroups:MTLSizeMake((NSUInteger)groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [normsEncoder endEncoding];

        id<MTLComputeCommandEncoder> stepEncoder = [commandBuffer computeCommandEncoder];

        if (stepEncoder == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [stepEncoder setComputePipelineState:stepPipeline];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)momentumRef offset:0 atIndex:2];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:3];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
        [stepEncoder setBytes:&count length:sizeof(count) atIndex:5];
        [stepEncoder setBytes:&groupCount length:sizeof(groupCount) atIndex:6];
        [stepEncoder setBytes:configBytes length:configBytesLen atIndex:7];

        NSUInteger threadWidth = [stepPipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [stepEncoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [stepEncoder endEncoding];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        [commandBuffer commit];

        return 0;
    }
}
