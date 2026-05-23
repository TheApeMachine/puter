#include "matmul.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_matmul_module_source = NULL;

void cuda_matmul_register_module_source(const char* source) {
    g_cuda_matmul_module_source = source;
}

const char* cuda_matmul_module_source(void) {
    return g_cuda_matmul_module_source;
}

void cuda_matmul_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_matmul_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_matmul_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_matmul_status_set(status, -6, "unknown CUDA matmul dtype");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

int cuda_matmul_dispatch_tiled(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    int hasBias,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_matmul_status_clear(status);

    if (rows == 0 || inner == 0 || cols == 0) {
        return 0;
    }

    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        cuda_matmul_status_set(status, -2, "nil CUDA matmul buffer");
        return -2;
    }

    if (hasBias && biasRef == NULL) {
        cuda_matmul_status_set(status, -2, "nil CUDA matmul bias buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_matmul_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_matmul_module_source();

    if (moduleSource == NULL) {
        cuda_matmul_status_set(status, -7, "CUDA matmul module source not registered");
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

    void* leftPtr = cuda_buffer_device_ptr(leftRef);
    void* rightPtr = cuda_buffer_device_ptr(rightRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* biasPtr = hasBias ? cuda_buffer_device_ptr(biasRef) : NULL;

    uint32_t gridX = (cols + 15u) / 16u;
    uint32_t gridY = (rows + 15u) / 16u;
    uint32_t sharedBytes = 256u * 2u * (uint32_t)sizeof(float);

    if (hasBias) {
        void* args[] = {&leftPtr, &rightPtr, &biasPtr, &outPtr, &rows, &inner, &cols};
        int launchCode = cuda_launch_grid(
            context,
            kernel,
            stream,
            gridX,
            gridY,
            1,
            16,
            16,
            1,
            sharedBytes,
            args,
            sizeof(args),
            status
        );

        if (launchCode != 0) {
            return launchCode;
        }
    }

    if (!hasBias) {
        void* args[] = {&leftPtr, &rightPtr, &outPtr, &rows, &inner, &cols};
        int launchCode = cuda_launch_grid(
            context,
            kernel,
            stream,
            gridX,
            gridY,
            1,
            16,
            16,
            1,
            sharedBytes,
            args,
            sizeof(args),
            status
        );

        if (launchCode != 0) {
            return launchCode;
        }
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
