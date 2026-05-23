#include "math.h"
#include "elementwise.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const char* metal_unary_math_operation_name(int operation) {
    switch (operation) {
    case MetalUnaryMathAbs: return "abs";
    case MetalUnaryMathNeg: return "neg";
    case MetalUnaryMathSqrt: return "sqrt";
    case MetalUnaryMathReLU: return "relu";
    default: return NULL;
    }
}

int metal_dispatch_unary_math(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_elementwise_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_elementwise_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        const char* operationName = metal_unary_math_operation_name(operation);

        if (operationName == NULL) {
            metal_elementwise_status_set(status, -6, "unknown unary math operation");
            return -6;
        }

        char kernelName[128];
        int nameCode = metal_elementwise_compose_kernel_name(
            kernelName,
            sizeof(kernelName),
            operationName,
            metal_elementwise_element_dtype_suffix(elementDType),
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_elementwise_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder(context, &commandBuffer);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder
            dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        metal_track_command_completion(context, commandBuffer, completionToken, NULL);
        metal_end_encoder(context, encoder, commandBuffer);

        return 0;
    }
}
