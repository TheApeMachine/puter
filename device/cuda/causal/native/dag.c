#include "dag.h"
#include "causal_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_dag_markov_factorization(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef conditionalsRef,
    CUDABufferRef parentsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    (void)parentsRef;

    if (conditionalsRef == NULL || scratchRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* conditionalsPtr = cuda_buffer_device_ptr(conditionalsRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* partialArgs[] = {&conditionalsPtr, &scratchPtr, &count};

    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* finalizeArgs[] = {&scratchPtr, &outPtr, &partialCount};

    return cuda_causal_dag_two_phase_launch(
        contextRef,
        elementDType,
        partialCount,
        partialArgs,
        sizeof(partialArgs),
        finalizeArgs,
        sizeof(finalizeArgs),
        completionToken,
        status
    );
}
