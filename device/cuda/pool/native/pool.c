#include "pool.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_vision_module_source = NULL;

void cuda_vision_register_module_source(const char* source) {
    g_cuda_vision_module_source = source;
}

const char* cuda_vision_module_source(void) {
    return g_cuda_vision_module_source;
}

static void cuda_vision_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_vision_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_vision_status_set(status, -6, "unknown CUDA pool dtype");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

int cuda_vision_dispatch_pool2d(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_vision_status_clear(status);

    if (batch == 0 || channels == 0 || outHeight == 0 || outWidth == 0) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_vision_status_set(status, -2, "nil CUDA pool buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_vision_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_vision_module_source();

    if (moduleSource == NULL) {
        cuda_vision_status_set(status, -7, "CUDA pool module source not registered");
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
    void* args[] = {
        &inputPtr,
        &outPtr,
        &batch,
        &channels,
        &inHeight,
        &inWidth,
        &outHeight,
        &outWidth,
    };
    uint32_t count = batch * channels * outHeight * outWidth;
    uint32_t launchCount = cuda_vector_launch_count(count, elementDType);
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
