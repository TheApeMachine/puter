#include "parametric.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>

int cuda_dispatch_unary_param(
    CUDADeviceRef contextRef,
    const char* operationPrefix,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    float param,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_activation_status_clear(status);

    if (count == 0 || operationPrefix == NULL) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_activation_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* dtypeSuffix = cuda_activation_element_dtype_suffix(elementDType);

    if (dtypeSuffix == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA parametric dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_activation_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationPrefix,
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
    void* args[] = {&inputPtr, &outputPtr, &count, &param};
    uint32_t launchCount = cuda_activation_vector_launch_count(count, elementDType);
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
