#include "free_energy.h"
#include "active_inference.h"
#include "../internal/bridge/core_private.h"

static int cuda_active_encode_free_energy_partial(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef likelihoodRef,
    CUDABufferRef posteriorRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    uint32_t count,
    uint32_t partialCount,
    CUDAStatus* status
) {
    void* likelihoodPtr = cuda_buffer_device_ptr(likelihoodRef);
    void* posteriorPtr = cuda_buffer_device_ptr(posteriorRef);
    void* priorPtr = cuda_buffer_device_ptr(priorRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* args[] = {&likelihoodPtr, &posteriorPtr, &priorPtr, &scratchPtr, &count};

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

static int cuda_active_encode_expected_partial(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef predictedObsRef,
    CUDABufferRef preferredObsRef,
    CUDABufferRef predictedStateRef,
    CUDABufferRef scratchRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    CUDAStatus* status
) {
    void* predictedObsPtr = cuda_buffer_device_ptr(predictedObsRef);
    void* preferredObsPtr = cuda_buffer_device_ptr(preferredObsRef);
    void* predictedStatePtr = cuda_buffer_device_ptr(predictedStateRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* args[] = {
        &predictedObsPtr,
        &preferredObsPtr,
        &predictedStatePtr,
        &scratchPtr,
        &obsCount,
        &stateCount,
        &obsPartialCount,
    };

    return cuda_launch_grid(
        context,
        kernel,
        stream,
        obsPartialCount + statePartialCount,
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

int cuda_dispatch_free_energy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef likelihoodRef,
    CUDABufferRef posteriorRef,
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

    if (likelihoodRef == NULL || posteriorRef == NULL || priorRef == NULL ||
        scratchRef == NULL || outRef == NULL) {
        cuda_active_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char partialName[128];
    char finalizeName[128];
    int partialNameCode = cuda_active_kernel_name(
        partialName, sizeof(partialName), "free_energy", "partial", elementDType, status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    int finalizeNameCode = cuda_active_kernel_name(
        finalizeName, sizeof(finalizeName), "active_scalar_finalize", "value", elementDType, status
    );

    if (finalizeNameCode != 0) {
        return finalizeNameCode;
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

    CUDAKernelRef finalizeKernel = cuda_get_kernel(context, cuda_active_module_source(), finalizeName, status);

    if (finalizeKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    int partialCode = cuda_active_encode_free_energy_partial(
        context,
        stream,
        partialKernel,
        likelihoodRef,
        posteriorRef,
        priorRef,
        scratchRef,
        count,
        partialCount,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    int finalizeCode = cuda_active_encode_finalize(
        context,
        stream,
        finalizeKernel,
        scratchRef,
        outRef,
        partialCount,
        status
    );

    if (finalizeCode != 0) {
        return finalizeCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_expected_free_energy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef predictedObsRef,
    CUDABufferRef preferredObsRef,
    CUDABufferRef predictedStateRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_active_status_clear(status);

    if (obsCount == 0 && stateCount == 0) {
        return 0;
    }

    if (predictedObsRef == NULL || preferredObsRef == NULL || predictedStateRef == NULL ||
        scratchRef == NULL || outRef == NULL) {
        cuda_active_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char partialName[128];
    char finalizeName[128];
    int partialNameCode = cuda_active_kernel_name(
        partialName, sizeof(partialName), "expected_free_energy", "partial", elementDType, status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    int finalizeNameCode = cuda_active_kernel_name(
        finalizeName, sizeof(finalizeName), "active_scalar_finalize", "value", elementDType, status
    );

    if (finalizeNameCode != 0) {
        return finalizeNameCode;
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

    CUDAKernelRef finalizeKernel = cuda_get_kernel(context, cuda_active_module_source(), finalizeName, status);

    if (finalizeKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    int partialCode = cuda_active_encode_expected_partial(
        context,
        stream,
        partialKernel,
        predictedObsRef,
        preferredObsRef,
        predictedStateRef,
        scratchRef,
        obsCount,
        stateCount,
        obsPartialCount,
        statePartialCount,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    uint32_t totalPartialCount = obsPartialCount + statePartialCount;
    int finalizeCode = cuda_active_encode_finalize(
        context,
        stream,
        finalizeKernel,
        scratchRef,
        outRef,
        totalPartialCount,
        status
    );

    if (finalizeCode != 0) {
        return finalizeCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
