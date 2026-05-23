#include "forward.h"
#include "predictive_coding.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_pc_prediction(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef weightsRef,
    CUDABufferRef stateRef,
    CUDABufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || outRef == NULL) {
        cuda_predictive_coding_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_predictive_coding_kernel_name(
        kernelName, sizeof(kernelName), "pc_prediction", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    void* weightsPtr = cuda_buffer_device_ptr(weightsRef);
    void* statePtr = cuda_buffer_device_ptr(stateRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&weightsPtr, &statePtr, &outPtr, &inCount};

    return cuda_predictive_coding_launch(
        contextRef,
        kernelName,
        outCount,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_pc_prediction_error(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef observedRef,
    CUDABufferRef predictedRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (observedRef == NULL || predictedRef == NULL || outRef == NULL) {
        cuda_predictive_coding_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_predictive_coding_kernel_name(
        kernelName, sizeof(kernelName), "pc_prediction_error", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    void* observedPtr = cuda_buffer_device_ptr(observedRef);
    void* predictedPtr = cuda_buffer_device_ptr(predictedRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&observedPtr, &predictedPtr, &outPtr, &count};

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
