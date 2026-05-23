#include "flash.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_flash_attention(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef queryRef,
    CUDABufferRef keyRef,
    CUDABufferRef valueRef,
    CUDABufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_attention_status_clear(status);

    if (queryRef == NULL || keyRef == NULL || valueRef == NULL || outRef == NULL) {
        cuda_transformer_status_set(status, -2, "nil CUDA flash attention buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_attention_kernel_name(
        kernelName,
        sizeof(kernelName),
        "flash_attention",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_attention_module_source();

    if (moduleSource == NULL) {
        cuda_transformer_status_set(status, -7, "CUDA attention module source not registered");
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

    void* queryPtr = cuda_buffer_device_ptr(queryRef);
    void* keyPtr = cuda_buffer_device_ptr(keyRef);
    void* valuePtr = cuda_buffer_device_ptr(valueRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&queryPtr, &keyPtr, &valuePtr, &outPtr, &seqQ, &seqK, &depth, &valueDim};
    int launchCode = cuda_launch_grid(
        context,
        kernel,
        stream,
        seqQ,
        (valueDim + 63u) / 64u,
        1u,
        256u,
        1u,
        1u,
        0u,
        args,
        sizeof(args),
        status
    );

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
