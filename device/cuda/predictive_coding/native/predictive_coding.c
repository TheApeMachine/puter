#include "predictive_coding.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_predictive_coding_module_source = NULL;

void cuda_predictive_coding_register_module_source(const char* source) {
    g_cuda_predictive_coding_module_source = source;
}

const char* cuda_predictive_coding_module_source(void) {
    return g_cuda_predictive_coding_module_source;
}

void cuda_predictive_coding_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_predictive_coding_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_predictive_coding_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_predictive_coding_status_set(status, -6, "unknown CUDA predictive coding kernel");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

static int cuda_predictive_coding_launch(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_predictive_coding_status_clear(status);

    if (launchCount == 0) {
        return 0;
    }

    const char* moduleSource = cuda_predictive_coding_module_source();

    if (moduleSource == NULL) {
        cuda_predictive_coding_status_set(status, -7, "CUDA predictive coding module source not registered");
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
