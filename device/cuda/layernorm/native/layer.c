#include "layer.h"
#include "../../internal/bridge/core_private.h"

#include <stdio.h>
#include <string.h>

#define cudaLayerNormThreadCount 256U

static const char* g_cuda_layernorm_module_source = NULL;

void cuda_layernorm_register_module_source(const char* source) {
    g_cuda_layernorm_module_source = source;
}

const char* cuda_layernorm_module_source(void) {
    return g_cuda_layernorm_module_source;
}

static void cuda_layernorm_status_clear(CUDAStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void cuda_layernorm_status_set(CUDAStatus* status, int code, const char* message) {
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

static int cuda_layernorm_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    CUDAStatus* status
) {
    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_layernorm_status_set(status, -6, "CUDA layernorm kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_layernorm_dispatch_grid(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef outputRef,
    uint32_t rows,
    uint32_t cols,
    int hasBias,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_layernorm_status_clear(status);

    if (rows == 0 || cols == 0) {
        return 0;
    }

    if (inputRef == NULL || scaleRef == NULL || outputRef == NULL) {
        cuda_layernorm_status_set(status, -2, "nil CUDA layernorm buffer");
        return -2;
    }

    if (hasBias && biasRef == NULL) {
        cuda_layernorm_status_set(status, -2, "nil CUDA layernorm bias buffer");
        return -2;
    }

    const char* dtypeSuffix = cuda_element_dtype_suffix(elementDType);

    if (dtypeSuffix == NULL) {
        cuda_layernorm_status_set(status, -6, "unknown CUDA layernorm dtype");
        return -6;
    }

    char kernelName[128];
    int nameCode = cuda_layernorm_compose_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        dtypeSuffix,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_layernorm_module_source();

    if (moduleSource == NULL) {
        cuda_layernorm_status_set(status, -7, "CUDA layernorm module source not registered");
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
    void* scalePtr = cuda_buffer_device_ptr(scaleRef);
    void* outputPtr = cuda_buffer_device_ptr(outputRef);
    void* biasPtr = hasBias ? cuda_buffer_device_ptr(biasRef) : NULL;

    if (hasBias) {
        void* args[] = {&inputPtr, &scalePtr, &biasPtr, &outputPtr, &cols};
        int launchCode = cuda_launch_grid(
            context,
            kernel,
            stream,
            rows,
            1,
            1,
            cudaLayerNormThreadCount,
            1,
            1,
            0,
            args,
            sizeof(args),
            status
        );

        if (launchCode != 0) {
            return launchCode;
        }
    }

    if (!hasBias) {
        void* args[] = {&inputPtr, &scalePtr, &outputPtr, &cols};
        int launchCode = cuda_launch_grid(
            context,
            kernel,
            stream,
            rows,
            1,
            1,
            cudaLayerNormThreadCount,
            1,
            1,
            0,
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

int cuda_dispatch_layernorm(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef outputRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_layernorm_dispatch_grid(
        contextRef,
        "layernorm",
        elementDType,
        inputRef,
        scaleRef,
        biasRef,
        outputRef,
        rows,
        cols,
        1,
        completionToken,
        status
    );
}

int cuda_dispatch_rmsnorm(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef outputRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_layernorm_dispatch_grid(
        contextRef,
        "rmsnorm",
        elementDType,
        inputRef,
        scaleRef,
        NULL,
        outputRef,
        rows,
        cols,
        0,
        completionToken,
        status
    );
}
