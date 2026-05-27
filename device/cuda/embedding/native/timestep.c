#include "timestep.h"
#include "embedding.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_timestep_embedding(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef timestepsRef,
    CUDABufferRef outRef,
    float maxPeriod,
    float downscaleFreqShift,
    float timestepDivisor,
    int flipSinToCos,
    uint32_t count,
    uint32_t dim,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_transformer_status_clear(status);

    if (count == 0 || dim == 0) {
        return 0;
    }

    if (
        timestepsRef == NULL ||
        outRef == NULL
    ) {
        cuda_transformer_status_set(status, -2, "nil CUDA timestep buffer");
        return -2;
    }

    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_transformer_status_set(status, -6, "unknown CUDA timestep dtype");
        return -6;
    }

    char kernelName[128];
    int written = snprintf(kernelName, sizeof(kernelName), "timestep_embedding_%s", suffix);

    if (written <= 0 || (size_t)written >= sizeof(kernelName)) {
        cuda_transformer_status_set(status, -6, "CUDA timestep kernel name overflow");
        return -6;
    }

    const char* moduleSource = cuda_embedding_module_source();

    if (moduleSource == NULL) {
        cuda_transformer_status_set(status, -7, "CUDA embedding module source not registered");
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

    void* timestepsPtr = cuda_buffer_device_ptr(timestepsRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {
        &timestepsPtr,
        &maxPeriod,
        &downscaleFreqShift,
        &timestepDivisor,
        &flipSinToCos,
        &outPtr,
        &count,
        &dim,
    };
    uint32_t launchCount = count * dim;
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
