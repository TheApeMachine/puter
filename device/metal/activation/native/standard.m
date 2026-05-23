#include "standard.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const char* metal_standard_unary_operation_name(int operation) {
    switch (operation) {
    case MetalUnaryFloat32Relu: return "relu";
    case MetalUnaryFloat32Abs: return "abs";
    case MetalUnaryFloat32Neg: return "neg";
    case MetalUnaryFloat32Square: return "square";
    case MetalUnaryFloat32Recip: return "recip";
    case MetalUnaryFloat32Sqrt: return "sqrt";
    case MetalUnaryFloat32Sign: return "sign";
    case MetalUnaryFloat32Rsqrt: return "rsqrt";
    case MetalUnaryFloat32Exp: return "exp";
    case MetalUnaryFloat32Log: return "log";
    case MetalUnaryFloat32Sin: return "sin";
    case MetalUnaryFloat32Cos: return "cos";
    case MetalUnaryFloat32Tanh: return "tanh";
    case MetalUnaryFloat32Gelu: return "gelu";
    case MetalUnaryFloat32Sigmoid: return "sigmoid";
    case MetalUnaryFloat32Silu: return "silu";
    case MetalUnaryFloat32Swish: return "swish";
    case MetalUnaryFloat32Softsign: return "softsign";
    case MetalUnaryFloat32ELU: return "elu";
    case MetalUnaryFloat32SELU: return "selu";
    case MetalUnaryFloat32LeakyReLU: return "leaky_relu";
    case MetalUnaryFloat32HardSigmoid: return "hardsigmoid";
    case MetalUnaryFloat32HardSwish: return "hardswish";
    case MetalUnaryFloat32Log1p: return "log1p";
    case MetalUnaryFloat32Expm1: return "expm1";
    case MetalUnaryFloat32CELU: return "celu";
    case MetalUnaryFloat32Softplus: return "softplus";
    case MetalUnaryFloat32Mish: return "mish";
    case MetalUnaryFloat32LogSigmoid: return "log_sigmoid";
    case MetalUnaryFloat32GeluTanh: return "gelu_tanh";
    case MetalUnaryFloat32HardTanh: return "hardtanh";
    case MetalUnaryFloat32HardGelu: return "hard_gelu";
    case MetalUnaryFloat32QuickGelu: return "quick_gelu";
    case MetalUnaryFloat32TanhShrink: return "tanh_shrink";
    default: return NULL;
    }
}

static int metal_standard_prepare(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    if (context == NULL || context->queue == NULL) {
        metal_activation_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_dispatch_unary_elementwise(
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
        metal_activation_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_activation_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        const char* operationName = metal_standard_unary_operation_name(operation);

        if (operationName == NULL) {
            metal_activation_status_set(status, -6, "unknown Metal standard unary operation");
            return -6;
        }

        const char* dtypeSuffix = metal_activation_element_dtype_suffix(elementDType);

        if (dtypeSuffix == NULL) {
            metal_activation_status_set(status, -6, "unknown Metal standard unary dtype");
            return -6;
        }

        char kernelName[128];
        int nameCode = metal_activation_compose_kernel_name(
            kernelName,
            sizeof(kernelName),
            operationName,
            dtypeSuffix,
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_standard_prepare(
            (MetalContext*)contextRef,
            kernelName,
            status,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

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
        metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
        metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

        return 0;
    }
}
