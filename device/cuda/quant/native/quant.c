#include "quant_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_quant_module_source = NULL;

void cuda_quant_register_module_source(const char* source) {
    g_cuda_quant_module_source = source;
}

const char* cuda_quant_module_source(void) {
    return g_cuda_quant_module_source;
}

int cuda_dispatch_quantization(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    float invScale,
    int zeroPoint,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA quant dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_compose_kernel_name(kernelName, sizeof(kernelName), "int8_quant", suffix, status);

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_quant_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA quant module source not registered");
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
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &outPtr, &invScale, &zeroPoint, &count};
    uint32_t launchCount = cuda_vector_launch_count(count, elementDType);
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
