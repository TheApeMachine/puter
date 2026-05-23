#include "matmul.h"

#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>

void metal_matmul_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_matmul_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_matmul_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_matmul_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_matmul_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_matmul_status_set(status, -6, "unknown Metal matmul kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_matmul_status_set(status, -6, "Metal matmul kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_matmul_mps_element_size(int elementDType, MetalStatus* status) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return (int)sizeof(float);
    case MetalElementDTypeFloat16: return (int)sizeof(uint16_t);
    case MetalElementDTypeBFloat16: return (int)sizeof(uint16_t);
    default:
        metal_matmul_status_set(status, -6, "unsupported Metal matmul dtype");
        return 0;
    }
}

static MPSDataType metal_matmul_mps_dtype(int elementDType, MetalStatus* status) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return MPSDataTypeFloat32;
    case MetalElementDTypeFloat16: return MPSDataTypeFloat16;
    case MetalElementDTypeBFloat16: return MPSDataTypeBFloat16;
    default:
        metal_matmul_status_set(status, -6, "unsupported MPS matmul dtype");
        return MPSDataTypeInvalid;
    }
}

static bool metal_matmul_mps_row_bytes_aligned(uint32_t columnCount, int elementSize) {
    NSUInteger rowBytes = (NSUInteger)columnCount * (NSUInteger)elementSize;

    return rowBytes % 256 == 0;
}

static int metal_matmul_dispatch_mps(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_matmul_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->device == NULL || context->queue == NULL) {
            metal_matmul_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        int elementSize = metal_matmul_mps_element_size(elementDType, status);

        if (elementSize == 0) {
            return status != NULL && status->code != 0 ? status->code : -6;
        }

        if (!metal_matmul_mps_row_bytes_aligned(inner, elementSize)
            || !metal_matmul_mps_row_bytes_aligned(cols, elementSize)) {
            return -100;
        }

        MPSDataType dataType = metal_matmul_mps_dtype(elementDType, status);

        if (dataType == MPSDataTypeInvalid) {
            return status != NULL && status->code != 0 ? status->code : -6;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        MPSMatrixDescriptor* leftDescriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rows
            columns:inner
            rowBytes:inner * (NSUInteger)elementSize
            dataType:dataType
        ];
        MPSMatrixDescriptor* rightDescriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:inner
            columns:cols
            rowBytes:cols * (NSUInteger)elementSize
            dataType:dataType
        ];
        MPSMatrixDescriptor* outDescriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rows
            columns:cols
            rowBytes:cols * (NSUInteger)elementSize
            dataType:dataType
        ];

        MPSMatrix* leftMatrix = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>)leftRef
            descriptor:leftDescriptor
        ];
        MPSMatrix* rightMatrix = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>)rightRef
            descriptor:rightDescriptor
        ];
        MPSMatrix* outMatrix = [[MPSMatrix alloc]
            initWithBuffer:(__bridge id<MTLBuffer>)outRef
            descriptor:outDescriptor
        ];

        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:NO
            resultRows:rows
            resultColumns:cols
            interiorColumns:inner
            alpha:1.0
            beta:0.0
        ];

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer;

        if (context->isBatching) {
            if (context->currentCommandBuffer == NULL) {
                id<MTLCommandBuffer> batchBuffer = [queue commandBuffer];
                context->currentCommandBuffer = (__bridge_retained void*)batchBuffer;
            }

            commandBuffer = (__bridge id<MTLCommandBuffer>)context->currentCommandBuffer;
            metal_suspend_compute_encoder(context);
        } else {
            commandBuffer = [queue commandBuffer];
        }

        [matmul encodeToCommandBuffer:commandBuffer leftMatrix:leftMatrix rightMatrix:rightMatrix resultMatrix:outMatrix];
        metal_track_command_completion(context, commandBuffer, completionToken, NULL);

        if (!context->isBatching) {
            [commandBuffer commit];
        }

        return 0;
    }
}

static int metal_matmul_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_matmul_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_matmul_status_set(status, -1, "invalid Metal context");
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
            dispatchThreadgroups:MTLSizeMake((cols + 15) / 16, (rows + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
        ];
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}

