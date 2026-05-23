#include "softmax.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>

static int cuda_softmax_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA softmax dtype");
        return -6;
    }

    return cuda_activation_compose_kernel_name(out, outBytes, "softmax", suffix, status);
}

int cuda_dispatch_softmax(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_activation_status_clear(status);

    if (inputRef == NULL || outRef == NULL) {
        cuda_activation_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    if (rows == 0 || cols == 0) {
        return 0;
    }

    char kernelName[128];
    int nameCode = cuda_softmax_kernel_name(
        kernelName,
        sizeof(kernelName),
        elementDType,
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
    void* args[] = {&inputPtr, &outputPtr, &cols};
    uint32_t blockSize = 256;
    uint32_t elementBytes = sizeof(float);

    switch (elementDType) {
    case CUDAElementDTypeFloat16:
        elementBytes = 2u;
        break;
    case CUDAElementDTypeBFloat16:
        elementBytes = 2u;
        break;
    default:
        elementBytes = sizeof(float);
        break;
    }

    uint32_t sharedBytes = blockSize * 2u * elementBytes;
    int launchCode = cuda_launch_grid(
        context,
        kernel,
        stream,
        rows,
        1,
        1,
        blockSize,
        1,
        1,
        sharedBytes,
        args,
        sizeof(args),
        status
    );

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
