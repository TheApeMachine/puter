#include "mask.h"
#include "dropout.h"
#include "../internal/bridge/core_private.h"

static size_t cuda_dropout_element_bytes(int elementDType) {
    switch (elementDType) {
    case CUDAElementDTypeFloat32:
        return sizeof(float);
    case CUDAElementDTypeFloat16:
        return sizeof(unsigned short);
    case CUDAElementDTypeBFloat16:
        return sizeof(unsigned short);
    default:
        return 0;
    }
}

int cuda_dispatch_dropout(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seedX,
    uint32_t seedY,
    uint32_t seedZ,
    uint32_t seedW,
    int identity,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_dropout_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_dropout_status_set(status, -2, "nil CUDA dropout buffer");
        return -2;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);

    if (identity) {
        size_t elementBytes = cuda_dropout_element_bytes(elementDType);

        if (elementBytes == 0) {
            cuda_dropout_status_set(status, -6, "unknown CUDA dropout dtype");
            return -6;
        }

        long long byteCount = (long long)count * (long long)elementBytes;
        int copyCode = cuda_memcpy_async_d2d(outRef, inputRef, byteCount, stream, status);

        if (copyCode != 0) {
            return copyCode;
        }

        cuda_track_completion(context, stream, completionToken, NULL, status);
        return 0;
    }

    char kernelName[128];
    int nameCode = cuda_dropout_kernel_name(
        kernelName,
        sizeof(kernelName),
        "dropout",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_dropout_module_source();

    if (moduleSource == NULL) {
        cuda_dropout_status_set(status, -7, "CUDA dropout module source not registered");
        return -7;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
        unsigned int w;
    } seed = {seedX, seedY, seedZ, seedW};

    void* args[] = {
        &inputPtr,
        &outPtr,
        &count,
        &scale,
        &threshold,
        &seed,
    };
    int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
