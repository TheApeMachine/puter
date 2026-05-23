#include "learning.h"
#include "predictive_coding.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_pc_update_representation(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef errorRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    float learningRate,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || errorRef == NULL || outRef == NULL) {
        cuda_predictive_coding_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_predictive_coding_kernel_name(
        kernelName, sizeof(kernelName), "pc_update_representation", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    void* weightsPtr = cuda_buffer_device_ptr(weightsRef);
    void* statePtr = cuda_buffer_device_ptr(stateRef);
    void* errorPtr = cuda_buffer_device_ptr(errorRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&weightsPtr, &statePtr, &errorPtr, &outPtr, &outCount, &inCount, &learningRate};

    return cuda_predictive_coding_launch(
        contextRef,
        kernelName,
        inCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_pc_update_weights(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef errorRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    float learningRate,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || errorRef == NULL || outRef == NULL) {
        cuda_predictive_coding_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_predictive_coding_kernel_name(
        kernelName, sizeof(kernelName), "pc_update_weights", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    uint32_t count = outCount * inCount;
    void* weightsPtr = cuda_buffer_device_ptr(weightsRef);
    void* statePtr = cuda_buffer_device_ptr(stateRef);
    void* errorPtr = cuda_buffer_device_ptr(errorRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&weightsPtr, &statePtr, &errorPtr, &outPtr, &inCount, &count, &learningRate};

    return cuda_predictive_coding_launch(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
