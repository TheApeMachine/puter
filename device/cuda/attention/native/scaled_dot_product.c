#include "scaled_dot_product.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

static int cuda_attention_get_kernel(
    CUDAContext* context,
    const char* operationName,
    int elementDType,
    CUDAKernelRef* kernel,
    CUDAStatus* status
) {
    char kernelName[128];
    int nameCode = cuda_attention_kernel_name(
        kernelName,
        sizeof(kernelName),
        operationName,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_attention_module_source();

    if (moduleSource == NULL) {
        cuda_attention_status_clear(status);
        cuda_transformer_status_set(status, -7, "CUDA attention module source not registered");
        return -7;
    }

    *kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (*kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int cuda_attention_launch_scores(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef queryRef,
    CUDABufferRef keyRef,
    CUDABufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    CUDAStatus* status
) {
    void* queryPtr = cuda_buffer_device_ptr(queryRef);
    void* keyPtr = cuda_buffer_device_ptr(keyRef);
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* args[] = {&queryPtr, &keyPtr, &scoresPtr, &seqQ, &seqK, &depth};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        (seqK + 15u) / 16u,
        (seqQ + 15u) / 16u,
        1u,
        16u,
        16u,
        1u,
        0u,
        args,
        sizeof(args),
        status
    );
}

static int cuda_attention_launch_softmax(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK,
    CUDAStatus* status
) {
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* args[] = {&scoresPtr, &seqK, &seqQ};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        seqQ,
        1u,
        1u,
        256u,
        1u,
        1u,
        0u,
        args,
        sizeof(args),
        status
    );
}

static int cuda_attention_launch_weighted(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    CUDABufferRef scoresRef,
    CUDABufferRef valueRef,
    CUDABufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t valueDim,
    CUDAStatus* status
) {
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* valuePtr = cuda_buffer_device_ptr(valueRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scoresPtr, &valuePtr, &outPtr, &seqQ, &seqK, &valueDim};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        (valueDim + 15u) / 16u,
        (seqQ + 15u) / 16u,
        1u,
        16u,
        16u,
        1u,
        0u,
        args,
        sizeof(args),
        status
    );
}

int cuda_dispatch_attention(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef queryRef,
    CUDABufferRef keyRef,
    CUDABufferRef valueRef,
    CUDABufferRef scoresRef,
    CUDABufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_attention_status_clear(status);

    if (queryRef == NULL || keyRef == NULL || valueRef == NULL ||
        scoresRef == NULL || outRef == NULL) {
        cuda_transformer_status_set(status, -2, "nil CUDA attention buffer");
        return -2;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef scoresKernel = NULL;
    int scoresPrepareCode = cuda_attention_get_kernel(
        context, "attention_scores", elementDType, &scoresKernel, status
    );

    if (scoresPrepareCode != 0) {
        return scoresPrepareCode;
    }

    CUDAKernelRef softmaxKernel = NULL;
    const char* moduleSource = cuda_attention_module_source();

    if (moduleSource == NULL) {
        cuda_transformer_status_set(status, -7, "CUDA attention module source not registered");
        return -7;
    }

    softmaxKernel = cuda_get_kernel(context, moduleSource, "attention_softmax", status);

    if (softmaxKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    CUDAKernelRef weightedKernel = NULL;
    int weightedPrepareCode = cuda_attention_get_kernel(
        context, "attention_weighted", elementDType, &weightedKernel, status
    );

    if (weightedPrepareCode != 0) {
        return weightedPrepareCode;
    }

    int scoresCode = cuda_attention_launch_scores(
        context, scoresKernel, stream,
        queryRef, keyRef, scoresRef,
        seqQ, seqK, depth, status
    );

    if (scoresCode != 0) {
        return scoresCode;
    }

    int softmaxCode = cuda_attention_launch_softmax(
        context, softmaxKernel, stream, scoresRef, seqQ, seqK, status
    );

    if (softmaxCode != 0) {
        return softmaxCode;
    }

    int weightedCode = cuda_attention_launch_weighted(
        context, weightedKernel, stream,
        scoresRef, valueRef, outRef,
        seqQ, seqK, valueDim, status
    );

    if (weightedCode != 0) {
        return weightedCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
