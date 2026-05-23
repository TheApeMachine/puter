#include "bag.h"
#include "embedding.h"
#include "../internal/bridge/core_private.h"

static int cuda_embedding_dispatch_bag(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef tableRef,
    CUDABufferRef indicesRef,
    CUDABufferRef offsetsRef,
    CUDABufferRef outRef,
    CUDABufferRef errorFlagRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    char kernelName[128];
    int nameCode = cuda_transformer_kernel_name(
        kernelName,
        sizeof(kernelName),
        "embedding_bag",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_embedding_module_source();

    if (moduleSource == NULL) {
        cuda_transformer_status_set(status, -7, "CUDA embedding module source not registered");
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

    void* tablePtr = cuda_buffer_device_ptr(tableRef);
    void* indicesPtr = cuda_buffer_device_ptr(indicesRef);
    void* offsetsPtr = cuda_buffer_device_ptr(offsetsRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* errorFlagPtr = cuda_buffer_device_ptr(errorFlagRef);
    void* args[] = {
        &tablePtr,
        &indicesPtr,
        &offsetsPtr,
        &outPtr,
        &errorFlagPtr,
        &vocab,
        &hidden,
        &indexCount,
        &bagCount,
    };
    uint32_t launchCount = bagCount * hidden;
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_embedding_bag(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef tableRef,
    CUDABufferRef indicesRef,
    CUDABufferRef offsetsRef,
    CUDABufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_embedding_dispatch_bag(
        contextRef,
        elementDType,
        tableRef,
        indicesRef,
        offsetsRef,
        outRef,
        NULL,
        vocab,
        hidden,
        indexCount,
        bagCount,
        completionToken,
        status
    );
}
