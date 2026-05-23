#include "physics.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_physics_module_source = NULL;

void cuda_physics_register_module_source(const char* source) {
    g_cuda_physics_module_source = source;
}

const char* cuda_physics_module_source(void) {
    return g_cuda_physics_module_source;
}

int cuda_physics_kernel_name(
    char* out,
    size_t outBytes,
    const char* operation,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operation == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA physics kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operation, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA physics kernel name overflow");
        return -6;
    }

    return 0;
}

int cuda_physics_prefixed_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    const char* suffix,
    CUDAStatus* status
) {
    const char* prefix = cuda_element_dtype_suffix(elementDType);

    if (prefix == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA physics FFT kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA physics FFT kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_physics_launch(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    const char* moduleSource = cuda_physics_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA physics module source not registered");
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

    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, argsBytes, status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_physics_launch_kernel(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_physics_launch(contextRef, kernelName, launchCount, args, argsBytes, completionToken, status);
}

int cuda_physics_launch_kernel_no_track(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    CUDAStatus* status
) {
    const char* moduleSource = cuda_physics_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA physics module source not registered");
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

    return cuda_launch_1d(context, kernel, stream, launchCount, args, argsBytes, status);
}
