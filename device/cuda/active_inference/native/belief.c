#include "belief.h"
#include "active_inference.h"
#include "../internal/bridge/core_private.h"

static int cuda_active_encode_belief_partial(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef likelihoodRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    uint32_t count,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* likelihoodPtr = cuda_buffer_device_ptr(likelihoodRef);
    void* priorPtr = cuda_buffer_device_ptr(priorRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* args[] = {&likelihoodPtr, &priorPtr, &scratchPtr, &count};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        partialCount,
        1,
        1,
        256,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}

static int cuda_active_encode_belief_normalize(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef likelihoodRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* likelihoodPtr = cuda_buffer_device_ptr(likelihoodRef);
    void* priorPtr = cuda_buffer_device_ptr(priorRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&likelihoodPtr, &priorPtr, &scratchPtr, &outPtr, &count, &partialCount};

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        1,
        1,
        1,
        256,
        1,
        1,
        0,
        args,
        sizeof(args),
        status
    );
}

int cuda_dispatch_belief_update(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef likelihoodRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_active_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (likelihoodRef == NULL || priorRef == NULL || scratchRef == NULL || outRef == NULL) {
        cuda_active_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char partialName[128];
    char normalizeName[128];
    int partialNameCode = cuda_active_kernel_name(
        partialName, sizeof(partialName), "belief_update", "partial", elementDType, status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    int normalizeNameCode = cuda_active_kernel_name(
        normalizeName, sizeof(normalizeName), "belief_update", "normalize", elementDType, status
    );

    if (normalizeNameCode != 0) {
        return normalizeNameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef partialKernel = NULL;
    int partialKernelCode = cuda_active_get_kernel(
        contextRef, partialName, status, &context, &stream, &partialKernel
    );

    if (partialKernelCode != 0) {
        return partialKernelCode;
    }

    CUDAKernelRef normalizeKernel = cuda_get_kernel(context, cuda_active_module_source(), normalizeName, status);

    if (normalizeKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    int partialCode = cuda_active_encode_belief_partial(
        context,
        stream,
        partialKernel,
        likelihoodRef,
        priorRef,
        scratchRef,
        count,
        partialCount,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    int normalizeCode = cuda_active_encode_belief_normalize(
        context,
        stream,
        normalizeKernel,
        likelihoodRef,
        priorRef,
        scratchRef,
        outRef,
        count,
        partialCount,
        status
    );

    if (normalizeCode != 0) {
        return normalizeCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_precision_weight(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef errorsRef,
    CUDABufferRef precisionRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_active_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (errorsRef == NULL || precisionRef == NULL || outRef == NULL) {
        cuda_active_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_active_single_kernel_name(
        kernelName, sizeof(kernelName), "precision_weight", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef kernel = NULL;
    int kernelCode = cuda_active_get_kernel(contextRef, kernelName, status, &context, &stream, &kernel);

    if (kernelCode != 0) {
        return kernelCode;
    }

    void* errorsPtr = cuda_buffer_device_ptr(errorsRef);
    void* precisionPtr = cuda_buffer_device_ptr(precisionRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&errorsPtr, &precisionPtr, &outPtr, &count};
    int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
