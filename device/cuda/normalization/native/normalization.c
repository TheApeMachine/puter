#include "normalization.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const uint32_t cudaNormalizationThreadCount = 256u;

static const char* g_cuda_normalization_module_source = NULL;

void cuda_normalization_register_module_source(const char* source) {
    g_cuda_normalization_module_source = source;
}

const char* cuda_normalization_module_source(void) {
    return g_cuda_normalization_module_source;
}

void cuda_normalization_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_normalization_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_normalization_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_normalization_status_set(status, -6, "unknown CUDA normalization dtype");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

int cuda_normalization_dispatch_rows(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    void** bufferRefs,
    size_t bufferCount,
    void** uintArgs,
    size_t uintArgCount,
    uint32_t rows,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_normalization_status_clear(status);

    if (rows == 0) {
        return 0;
    }

    char kernelName[128];
    int nameCode = cuda_normalization_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_normalization_module_source();

    if (moduleSource == NULL) {
        cuda_normalization_status_set(status, -7, "CUDA normalization module source not registered");
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

    void* args[16];
    size_t argIndex = 0;

    for (size_t bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
        args[argIndex++] = bufferRefs[bufferIndex];
    }

    for (size_t uintIndex = 0; uintIndex < uintArgCount; uintIndex++) {
        args[argIndex++] = uintArgs[uintIndex];
    }

    int launchCode = cuda_launch_grid(
        context,
        kernel,
        stream,
        rows,
        1,
        1,
        cudaNormalizationThreadCount,
        1,
        1,
        0,
        args,
        argIndex * sizeof(void*),
        status
    );

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
