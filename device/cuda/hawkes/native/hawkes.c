#include "hawkes_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_hawkes_module_source = NULL;

static const unsigned int cudaHMThreadCount = 256U;

void cuda_hawkes_register_module_source(const char* source) {
    g_cuda_hawkes_module_source = source;
}

const char* cuda_hawkes_module_source(void) {
    return g_cuda_hawkes_module_source;
}

static void cuda_hm_status_clear(CUDAStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void cuda_hm_status_set(CUDAStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, CUDA_STATUS_MESSAGE_BYTES, "%s", message);
}

int cuda_hm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_hm_status_set(status, -6, "unknown CUDA Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_hm_status_set(status, -6, "CUDA Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

int cuda_hm_phase_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        cuda_hm_status_set(status, -6, "unknown CUDA Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_hm_status_set(status, -6, "CUDA Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_hm_get_kernel(
    CUDADeviceRef contextRef,
    const char* kernelName,
    CUDAStatus* status,
    CUDAContext** context,
    CUDAStreamRef* stream,
    CUDAKernelRef* kernel
) {
    const char* moduleSource = cuda_hawkes_module_source();

    if (moduleSource == NULL) {
        cuda_hm_status_set(status, -7, "CUDA Hawkes module source not registered");
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

int cuda_hm_encode_hawkes_log_partial(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef eventsRef,
    CUDABufferRef totalTimeRef,
    CUDABufferRef baselineRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef scratchRef,
    uint32_t eventCount,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* eventsPtr = cuda_buffer_device_ptr(eventsRef);
    void* totalTimePtr = cuda_buffer_device_ptr(totalTimeRef);
    void* baselinePtr = cuda_buffer_device_ptr(baselineRef);
    void* alphaPtr = cuda_buffer_device_ptr(alphaRef);
    void* betaPtr = cuda_buffer_device_ptr(betaRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* args[] = {
        &eventsPtr,
        &totalTimePtr,
        &baselinePtr,
        &alphaPtr,
        &betaPtr,
        &scratchPtr,
        &eventCount,
    };

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        partialCount,
        1,
        1,
        cudaHMThreadCount,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}

int cuda_hm_encode_hawkes_log_finalize(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef scratchRef,
    CUDABufferRef totalTimeRef,
    CUDABufferRef baselineRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    CUDAStatus* status
) {
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* totalTimePtr = cuda_buffer_device_ptr(totalTimeRef);
    void* baselinePtr = cuda_buffer_device_ptr(baselineRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scratchPtr, &totalTimePtr, &baselinePtr, &outPtr, &eventCount};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        1,
        1,
        1,
        cudaHMThreadCount,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}

int cuda_hm_encode_mi_partial(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef jointRef,
    CUDABufferRef scratchRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* jointPtr = cuda_buffer_device_ptr(jointRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* args[] = {&jointPtr, &scratchPtr, &rows, &cols};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        partialCount,
        1,
        1,
        cudaHMThreadCount,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}

int cuda_hm_encode_finalize(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
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
        cudaHMThreadCount,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}
