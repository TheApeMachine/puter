#include "active_inference.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const unsigned int cudaActiveThreadCount = 256u;

static const char* g_cuda_active_module_source = NULL;

void cuda_active_register_module_source(const char* source) {
    g_cuda_active_module_source = source;
}

const char* cuda_active_module_source(void) {
    return g_cuda_active_module_source;
}

void cuda_active_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_active_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_active_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        cuda_active_status_set(status, -6, "unknown CUDA active inference kernel");
        return -6;
    }

    char prefix[128];
    int written = snprintf(prefix, sizeof(prefix), "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= sizeof(prefix)) {
        cuda_active_status_set(status, -6, "CUDA active inference kernel name overflow");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, prefix, phase, status);
}

int cuda_active_single_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_active_status_set(status, -6, "unknown CUDA active inference kernel");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

static int cuda_active_get_kernel(
    CUDADeviceRef contextRef,
    const char* kernelName,
    CUDAStatus* status,
    CUDAContext** context,
    CUDAStreamRef* stream,
    CUDAKernelRef* kernel
) {
    const char* moduleSource = cuda_active_module_source();

    if (moduleSource == NULL) {
        cuda_active_status_set(status, -7, "CUDA active inference module source not registered");
        return -7;
    }

    int prepareCode = cuda_context_prepare(contextRef, status, context, stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    *kernel = cuda_get_kernel(*context, moduleSource, kernelName, status);

    if (*kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int cuda_active_encode_finalize(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scratchPtr, &outPtr, &partialCount};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        1,
        1,
        1,
        cudaActiveThreadCount,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}
