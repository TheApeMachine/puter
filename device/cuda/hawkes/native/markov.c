#include "markov.h"
#include "hawkes_dispatch.h"
#include "../../internal/bridge/core_private.h"

int cuda_dispatch_markov_blanket_partition(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef adjacencyRef,
    CUDABufferRef internalRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    uint32_t internalCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (nodeCount == 0) {
        return 0;
    }

    if (adjacencyRef == NULL || internalRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA Markov blanket partition buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_hm_kernel_name(
        kernelName,
        sizeof(kernelName),
        "markov_blanket_partition",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef kernel = NULL;
    int prepareCode = cuda_hm_get_kernel(contextRef, kernelName, status, &context, &stream, &kernel);

    if (prepareCode != 0) {
        return prepareCode;
    }

    void* adjacencyPtr = cuda_buffer_device_ptr(adjacencyRef);
    void* internalPtr = cuda_buffer_device_ptr(internalRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&adjacencyPtr, &internalPtr, &outPtr, &nodeCount, &internalCount};
    int launchCode = cuda_launch_1d(context, kernel, stream, nodeCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_markov_flow(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef mutualInformationRef,
    CUDABufferRef partitionRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    int32_t targetLabel,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (nodeCount == 0) {
        return 0;
    }

    if (mutualInformationRef == NULL || partitionRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA Markov flow buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_hm_kernel_name(kernelName, sizeof(kernelName), "markov_flow", elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef kernel = NULL;
    int prepareCode = cuda_hm_get_kernel(contextRef, kernelName, status, &context, &stream, &kernel);

    if (prepareCode != 0) {
        return prepareCode;
    }

    void* mutualInformationPtr = cuda_buffer_device_ptr(mutualInformationRef);
    void* partitionPtr = cuda_buffer_device_ptr(partitionRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&mutualInformationPtr, &partitionPtr, &outPtr, &nodeCount, &targetLabel};
    int launchCode = cuda_launch_1d(context, kernel, stream, nodeCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_markov_mutual_information(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef jointRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (rows == 0 || cols == 0) {
        return 0;
    }

    if (jointRef == NULL || scratchRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA Markov mutual information buffer");
        return -2;
    }

    char partialName[128];
    char finalizeName[128];
    int partialNameCode = cuda_hm_phase_kernel_name(
        partialName,
        sizeof(partialName),
        "markov_mutual_information",
        "partial",
        elementDType,
        status
    );
    int finalizeNameCode = cuda_hm_kernel_name(
        finalizeName,
        sizeof(finalizeName),
        "hawkes_markov_finalize",
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

    int partialCode = cuda_hm_encode_mi_partial(
        context,
        partialKernel,
        stream,
        jointRef,
        scratchRef,
        rows,
        cols,
        partialCount,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    int finalizeCode = cuda_hm_encode_finalize(
        context,
        finalizeKernel,
        stream,
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
