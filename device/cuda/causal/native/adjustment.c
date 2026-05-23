#include "adjustment.h"
#include "causal_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_backdoor_adjustment(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef conditionalRef,
    CUDABufferRef marginalRef,
    CUDABufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (conditionalRef == NULL || marginalRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* conditionalPtr = cuda_buffer_device_ptr(conditionalRef);
    void* marginalPtr = cuda_buffer_device_ptr(marginalRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&conditionalPtr, &marginalPtr, &outPtr, &xCount, &zCount, &yCount};

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "backdoor_adjustment",
        xCount * yCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_frontdoor_adjustment(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef mediatorRef,
    CUDABufferRef outcomeRef,
    CUDABufferRef marginalRef,
    CUDABufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (mediatorRef == NULL || outcomeRef == NULL || marginalRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* mediatorPtr = cuda_buffer_device_ptr(mediatorRef);
    void* outcomePtr = cuda_buffer_device_ptr(outcomeRef);
    void* marginalPtr = cuda_buffer_device_ptr(marginalRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&mediatorPtr, &outcomePtr, &marginalPtr, &outPtr, &xCount, &mCount, &yCount};

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "frontdoor_adjustment",
        xCount * yCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
