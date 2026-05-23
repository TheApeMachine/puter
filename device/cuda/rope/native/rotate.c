#include "rotate.h"
#include "rope.h"
#include "../internal/bridge/core_private.h"

static int cuda_rope_launch(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t launchCount,
    void** args,
    size_t argBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_rope_status_clear(status);

    if (launchCount == 0) {
        return 0;
    }

    if (inputRef == NULL || outRef == NULL) {
        cuda_rope_status_set(status, -2, "nil CUDA rope buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_rope_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_rope_module_source();

    if (moduleSource == NULL) {
        cuda_rope_status_set(status, -7, "CUDA rope module source not registered");
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

    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, argBytes, status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_rope(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    float theta,
    float ropeFactor,
    float lowFreqFactor,
    float highFreqFactor,
    uint32_t originalContext,
    uint32_t halfMode,
    uint32_t positionOffset,
    uint64_t completionToken,
    CUDAStatus* status
) {
    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {
        &inputPtr,
        &outPtr,
        &seqLen,
        &numHeads,
        &headDim,
        &pairCount,
        &theta,
        &ropeFactor,
        &lowFreqFactor,
        &highFreqFactor,
        &originalContext,
        &halfMode,
        &positionOffset,
    };

    return cuda_rope_launch(
        contextRef,
        "rope",
        elementDType,
        inputRef,
        outRef,
        pairCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_rope_pairs(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    CUDABufferRef cosRef,
    CUDABufferRef sinRef,
    uint32_t halfDim,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (cosRef == NULL || sinRef == NULL) {
        cuda_rope_status_set(status, -2, "nil CUDA rope pairs buffer");
        return -2;
    }

    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* cosPtr = cuda_buffer_device_ptr(cosRef);
    void* sinPtr = cuda_buffer_device_ptr(sinRef);
    void* args[] = {&outPtr, &inputPtr, &cosPtr, &sinPtr, &halfDim};

    return cuda_rope_launch(
        contextRef,
        "rope_pairs",
        elementDType,
        inputRef,
        outRef,
        halfDim,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
