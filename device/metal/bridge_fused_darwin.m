#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_fused_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal fused command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static const char* metal_fused_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_dispatch_unary_param(
    MetalDeviceRef contextRef,
    const char* kernelName,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float param,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (status != NULL) {
            status->code = 0;
            status->message[0] = '\0';
        }

        if (count == 0 || kernelName == NULL) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || inputRef == NULL || outRef == NULL) {
            if (status != NULL) {
                status->code = -2;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "invalid Metal unary param dispatch");
            }

            return -2;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&param length:sizeof(param) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_fused_complete(completionToken, completedBuffer);
        }];
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_axpy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef yRef,
    MetalBufferRef xRef,
    uint32_t count,
    float alpha,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (status != NULL) {
            status->code = 0;
            status->message[0] = '\0';
        }

        if (count == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || yRef == NULL || xRef == NULL) {
            if (status != NULL) {
                status->code = -2;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "invalid Metal axpy dispatch");
            }

            return -2;
        }

        const char* dtypeSuffix = metal_fused_dtype_suffix(elementDType);

        if (dtypeSuffix == NULL) {
            if (status != NULL) {
                status->code = -6;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "unsupported Metal axpy dtype");
            }

            return -6;
        }

        char axpyKernelName[64];
        int written = snprintf(axpyKernelName, sizeof(axpyKernelName), "axpy_%s", dtypeSuffix);

        if (written <= 0 || (size_t)written >= sizeof(axpyKernelName)) {
            if (status != NULL) {
                status->code = -6;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "Metal axpy kernel name overflow");
            }

            return -6;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, axpyKernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)yRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&alpha length:sizeof(alpha) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_fused_complete(completionToken, completedBuffer);
        }];
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_dot(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (status != NULL) {
            status->code = 0;
            status->message[0] = '\0';
        }

        if (count == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || leftRef == NULL || rightRef == NULL || outRef == NULL) {
            if (status != NULL) {
                status->code = -2;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "invalid Metal dot dispatch");
            }

            return -2;
        }

        const char* dtypeSuffix = metal_fused_dtype_suffix(elementDType);

        if (dtypeSuffix == NULL) {
            if (status != NULL) {
                status->code = -6;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "unsupported Metal dot dtype");
            }

            return -6;
        }

        char dotKernelName[64];
        int written = snprintf(dotKernelName, sizeof(dotKernelName), "dot_%s", dtypeSuffix);

        if (written <= 0 || (size_t)written >= sizeof(dotKernelName)) {
            if (status != NULL) {
                status->code = -6;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "Metal dot kernel name overflow");
            }

            return -6;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, dotKernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];

        NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadgroups = (count + maxThreads - 1) / maxThreads;
        if (threadgroups > 1024) { threadgroups = 1024; } // Cap threadgroups to avoid excessive atomic contention
        
        [encoder dispatchThreads:MTLSizeMake(threadgroups * maxThreads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(maxThreads, 1, 1)];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_fused_complete(completionToken, completedBuffer);
        }];
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

int metal_dispatch_cholesky(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t order,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (status != NULL) {
            status->code = 0;
            status->message[0] = '\0';
        }

        if (order == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || inputRef == NULL || outRef == NULL) {
            if (status != NULL) {
                status->code = -2;
                snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "invalid Metal cholesky dispatch");
            }

            return -2;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);
        
        // MPSMatrixDecompositionCholesky requires the command buffer, not the encoder.
        // We must end the current encoder before calling MPS.
        if (encoder != nil) {
            [encoder endEncoding];
        }

        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:order columns:order rowBytes:order * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)inputRef descriptor:desc];
        MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)outRef descriptor:desc];

        MPSMatrixDecompositionCholesky* cholesky = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:(__bridge id<MTLDevice>)context->device lower:YES order:order];

        [cholesky encodeToCommandBuffer:commandBuffer sourceMatrix:inputMatrix resultMatrix:outMatrix status:nil];

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_fused_complete(completionToken, completedBuffer);
        }];
        
        // We pass nil for encoder since we already ended it
        metal_end_encoder((MetalContext*)contextRef, nil, commandBuffer);

        return 0;
    }
}
