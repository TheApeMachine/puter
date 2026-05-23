#include "intervention.h"
#include "causal_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_do_intervene(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef adjacencyRef,
    CUDABufferRef intervenedRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    uint32_t intervenedCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (adjacencyRef == NULL || intervenedRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* adjacencyPtr = cuda_buffer_device_ptr(adjacencyRef);
    void* intervenedPtr = cuda_buffer_device_ptr(intervenedRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&adjacencyPtr, &intervenedPtr, &outPtr, &nodeCount, &intervenedCount};

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "do_intervene",
        nodeCount * nodeCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_cate(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef treatedRef,
    CUDABufferRef controlRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (treatedRef == NULL || controlRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* treatedPtr = cuda_buffer_device_ptr(treatedRef);
    void* controlPtr = cuda_buffer_device_ptr(controlRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&treatedPtr, &controlPtr, &outPtr, &count};

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "cate",
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_counterfactual(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef observedYRef,
    CUDABufferRef observedXRef,
    CUDABufferRef counterfactualXRef,
    CUDABufferRef slopeRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (observedYRef == NULL || observedXRef == NULL || counterfactualXRef == NULL ||
        slopeRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* observedYPtr = cuda_buffer_device_ptr(observedYRef);
    void* observedXPtr = cuda_buffer_device_ptr(observedXRef);
    void* counterfactualXPtr = cuda_buffer_device_ptr(counterfactualXRef);
    void* slopePtr = cuda_buffer_device_ptr(slopeRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {
        &observedYPtr,
        &observedXPtr,
        &counterfactualXPtr,
        &slopePtr,
        &outPtr,
        &count,
    };

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "counterfactual",
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_iv_estimate(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef instrumentRef,
    CUDABufferRef treatmentRef,
    CUDABufferRef outcomeRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (instrumentRef == NULL || treatmentRef == NULL || outcomeRef == NULL ||
        scratchRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* instrumentPtr = cuda_buffer_device_ptr(instrumentRef);
    void* treatmentPtr = cuda_buffer_device_ptr(treatmentRef);
    void* outcomePtr = cuda_buffer_device_ptr(outcomeRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* partialArgs[] = {&instrumentPtr, &treatmentPtr, &outcomePtr, &scratchPtr, &count};

    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* finalizeArgs[] = {&scratchPtr, &outPtr, &count, &partialCount};

    return cuda_causal_two_phase_launch(
        contextRef,
        elementDType,
        "iv_estimate",
        partialCount,
        partialArgs,
        sizeof(partialArgs),
        finalizeArgs,
        sizeof(finalizeArgs),
        completionToken,
        status
    );
}
