#include "likelihood.h"
#include "hawkes_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_hawkes_log_likelihood(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef eventsRef,
    CUDABufferRef totalTimeRef,
    CUDABufferRef baselineRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (eventCount == 0) {
        return 0;
    }

    if (
        eventsRef == NULL ||
        totalTimeRef == NULL ||
        baselineRef == NULL ||
        alphaRef == NULL ||
        betaRef == NULL ||
        scratchRef == NULL ||
        outRef == NULL
    ) {
        cuda_status_set(status, -2, "nil CUDA Hawkes log likelihood buffer");
        return -2;
    }

    char partialName[128];
    char finalizeName[128];
    int partialNameCode = cuda_hm_phase_kernel_name(
        partialName,
        sizeof(partialName),
        "hawkes_log_likelihood",
        "partial",
        elementDType,
        status
    );
    int finalizeNameCode = cuda_hm_phase_kernel_name(
        finalizeName,
        sizeof(finalizeName),
        "hawkes_log_likelihood",
        "finalize",
        elementDType,
        status
    );

    if (partialNameCode != 0 || finalizeNameCode != 0) {
        return partialNameCode != 0 ? partialNameCode : finalizeNameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef partialKernel = NULL;
    int prepareCode = cuda_hm_get_kernel(contextRef, partialName, status, &context, &stream, &partialKernel);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef finalizeKernel = cuda_get_kernel(context, cuda_hawkes_module_source(), finalizeName, status);

    if (finalizeKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    int partialCode = cuda_hm_encode_hawkes_log_partial(
        context,
        partialKernel,
        stream,
        eventsRef,
        totalTimeRef,
        baselineRef,
        alphaRef,
        betaRef,
        scratchRef,
        eventCount,
        partialCount,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    int finalizeCode = cuda_hm_encode_hawkes_log_finalize(
        context,
        finalizeKernel,
        stream,
        scratchRef,
        totalTimeRef,
        baselineRef,
        outRef,
        eventCount,
        status
    );

    if (finalizeCode != 0) {
        return finalizeCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
