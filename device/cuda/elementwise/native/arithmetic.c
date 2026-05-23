#include "arithmetic.h"
#include "elementwise.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>

static const char* cuda_binary_operation_name(int operation) {
    switch (operation) {
    case CUDABinaryFloat32Add:
        return "add";
    case CUDABinaryFloat32Sub:
        return "sub";
    case CUDABinaryFloat32Mul:
        return "mul";
    case CUDABinaryFloat32Div:
        return "div";
    case CUDABinaryFloat32Max:
        return "max";
    case CUDABinaryFloat32Min:
        return "min";
    default:
        return NULL;
    }
}

int cuda_dispatch_binary_elementwise(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_elementwise_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        cuda_elementwise_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* operationName = cuda_binary_operation_name(operation);

    if (operationName == NULL) {
        cuda_elementwise_status_set(status, -6, "unknown binary elementwise operation");
        return -6;
    }

    const char* dtypeSuffix = cuda_elementwise_element_dtype_suffix(elementDType);

    if (dtypeSuffix == NULL) {
        cuda_elementwise_status_set(status, -6, "unknown binary elementwise dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_elementwise_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        dtypeSuffix,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_elementwise_module_source();

    if (moduleSource == NULL) {
        cuda_elementwise_status_set(status, -7, "CUDA elementwise module source not registered");
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

    void* leftPtr = cuda_buffer_device_ptr(leftRef);
    void* rightPtr = cuda_buffer_device_ptr(rightRef);
    void* outputPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&leftPtr, &rightPtr, &outputPtr, &count};
    uint32_t launchCount = cuda_vector_launch_count(count, elementDType);
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
