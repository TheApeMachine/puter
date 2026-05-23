#include "masking.h"
#include "attention.h"
#include "../internal/bridge/core_private.h"

static int cuda_masking_launch_1d(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    uint32_t launchCount,
    void** args,
    size_t argBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_attention_status_clear(status);

    if (launchCount == 0) {
        return 0;
    }

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

    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, argBytes, status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_apply_mask(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef maskRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (inputRef == NULL || maskRef == NULL || outRef == NULL) {
        cuda_transformer_status_set(status, -2, "nil CUDA apply_mask buffer");
        return -2;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* maskPtr = cuda_buffer_device_ptr(maskRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &maskPtr, &outPtr, &count};
    uint32_t launchCount = cuda_vector_launch_count(count, elementDType);

    return cuda_masking_launch_1d(
        contextRef,
        "apply_mask",
        elementDType,
        launchCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_causal_mask(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (outRef == NULL) {
        cuda_transformer_status_set(status, -2, "nil CUDA causal_mask buffer");
        return -2;
    }

    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&outPtr, &rows, &cols};

    return cuda_masking_launch_1d(
        contextRef,
        "causal_mask",
        elementDType,
        rows,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_alibi_bias(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef scoresRef,
    CUDABufferRef slopeRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (scoresRef == NULL || slopeRef == NULL || outRef == NULL) {
        cuda_transformer_status_set(status, -2, "nil CUDA alibi_bias buffer");
        return -2;
    }

    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* slopePtr = cuda_buffer_device_ptr(slopeRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scoresPtr, &slopePtr, &outPtr, &rows, &cols};

    return cuda_masking_launch_1d(
        contextRef,
        "alibi_bias",
        elementDType,
        rows,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
