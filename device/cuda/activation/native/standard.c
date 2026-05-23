#include "standard.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>
#include <string.h>

static const char* cuda_standard_unary_operation_name(int operation) {
    switch (operation) {
    case CUDAUnaryFloat32Relu:
        return "relu";
    case CUDAUnaryFloat32Exp:
        return "exp";
    case CUDAUnaryFloat32Log:
        return "log";
    case CUDAUnaryFloat32Tanh:
        return "tanh";
    case CUDAUnaryFloat32Gelu:
        return "gelu";
    case CUDAUnaryFloat32Sigmoid:
        return "sigmoid";
    case CUDAUnaryFloat32Silu:
        return "silu";
    case CUDAUnaryFloat32Swish:
        return "swish";
    case CUDAUnaryFloat32Softsign:
        return "softsign";
    case CUDAUnaryFloat32ELU:
        return "elu";
    case CUDAUnaryFloat32SELU:
        return "selu";
    case CUDAUnaryFloat32LeakyReLU:
        return "leaky_relu";
    case CUDAUnaryFloat32HardSigmoid:
        return "hardsigmoid";
    case CUDAUnaryFloat32HardSwish:
        return "hardswish";
    case CUDAUnaryFloat32Log1p:
        return "log1p";
    case CUDAUnaryFloat32Expm1:
        return "expm1";
    case CUDAUnaryFloat32CELU:
        return "celu";
    case CUDAUnaryFloat32Softplus:
        return "softplus";
    case CUDAUnaryFloat32Mish:
        return "mish";
    case CUDAUnaryFloat32LogSigmoid:
        return "log_sigmoid";
    case CUDAUnaryFloat32GeluTanh:
        return "gelu_tanh";
    case CUDAUnaryFloat32HardTanh:
        return "hardtanh";
    case CUDAUnaryFloat32HardGelu:
        return "hard_gelu";
    case CUDAUnaryFloat32QuickGelu:
        return "quick_gelu";
    case CUDAUnaryFloat32TanhShrink:
        return "tanh_shrink";
    default:
        return NULL;
    }
}

int cuda_dispatch_unary_elementwise(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_activation_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_activation_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* operationName = cuda_standard_unary_operation_name(operation);

    if (operationName == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA standard unary operation");
        return -6;
    }

    const char* dtypeSuffix = cuda_activation_element_dtype_suffix(elementDType);

    if (dtypeSuffix == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA standard unary dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_activation_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        dtypeSuffix,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_activation_module_source();

    if (moduleSource == NULL) {
        cuda_activation_status_set(status, -7, "CUDA activation module source not registered");
        return -7;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* outputPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &outputPtr, &count};
    int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
