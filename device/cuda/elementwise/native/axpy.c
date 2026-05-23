#include "axpy.h"
#include "elementwise.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>

int cuda_dispatch_axpy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef yRef,
    CUDABufferRef xRef,
    float alpha,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_elementwise_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (yRef == NULL || xRef == NULL) {
        cuda_elementwise_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* dtypeSuffix = cuda_elementwise_element_dtype_suffix(elementDType);

    if (dtypeSuffix == NULL) {
        cuda_elementwise_status_set(status, -6, "unknown axpy dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_elementwise_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        "axpy",
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

    void* yPtr = cuda_buffer_device_ptr(yRef);
    void* xPtr = cuda_buffer_device_ptr(xRef);
    void* args[] = {&yPtr, &xPtr, &count, &alpha};
    int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
