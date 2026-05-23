#include "vsa_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_vsa_module_source = NULL;

void cuda_vsa_register_module_source(const char* source) {
    g_cuda_vsa_module_source = source;
}

const char* cuda_vsa_module_source(void) {
    return g_cuda_vsa_module_source;
}

static const char* cuda_vsa_operation_name(int operation) {
    switch (operation) {
    case 0: return "vsa_bind";
    case 1: return "vsa_bundle";
    case 2: return "vsa_permute";
    case 3: return "vsa_inverse_permute";
    default: return NULL;
    }
}

static int cuda_vsa_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA VSA kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA VSA kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_vsa_launch(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    const char* moduleSource = cuda_vsa_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA VSA module source not registered");
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

int cuda_dispatch_vsa_binary(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* operationName = cuda_vsa_operation_name(operation);

    if (operationName == NULL) {
        cuda_status_set(status, -6, "unknown CUDA VSA binary operation");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_vsa_kernel_name(kernelName, sizeof(kernelName), operationName, elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    void* leftPtr = cuda_buffer_device_ptr(leftRef);
    void* rightPtr = cuda_buffer_device_ptr(rightRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&leftPtr, &rightPtr, &outPtr, &count};
    uint32_t launchCount = cuda_vector_launch_count(count, elementDType);

    return cuda_vsa_launch(
        contextRef,
        kernelName,
        launchCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_vsa_unary(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
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

    const char* operationName = cuda_vsa_operation_name(operation);

    if (operationName == NULL) {
        cuda_status_set(status, -6, "unknown CUDA VSA unary operation");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_vsa_kernel_name(kernelName, sizeof(kernelName), operationName, elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &outPtr, &count};

    return cuda_vsa_launch(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
