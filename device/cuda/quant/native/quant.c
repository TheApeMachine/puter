#include "quant_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_quant_module_source = NULL;

void cuda_quant_register_module_source(const char* source) {
    g_cuda_quant_module_source = source;
}

const char* cuda_quant_module_source(void) {
    return g_cuda_quant_module_source;
}

static const char* cuda_quantization_kernel_name(int operation) {
    switch (operation) {
    case 0: return "int8_dequant";
    case 1: return "int4_dequant";
    case 2: return "int8_quant";
    default: return NULL;
    }
}

int cuda_dispatch_quantization(
    CUDADeviceRef contextRef,
    int operation,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    float scale,
    int zeroPoint,
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

    const char* kernelName = cuda_quantization_kernel_name(operation);

    if (kernelName == NULL) {
        cuda_status_set(status, -6, "unknown CUDA quantization kernel");
        return -6;
    }

    const char* moduleSource = cuda_quant_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA quant module source not registered");
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

    if (operation == 2) {
        void* args[] = {&inputPtr, &outPtr, &count};
        int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

        if (launchCode != 0) {
            return launchCode;
        }
    }

    if (operation != 2) {
        void* args[] = {&outPtr, &inputPtr, &scale, &zeroPoint, &count};
        int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

        if (launchCode != 0) {
            return launchCode;
        }
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
