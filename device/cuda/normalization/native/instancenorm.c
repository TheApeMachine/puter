#include "instancenorm.h"
#include "normalization.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_instancenorm(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef outputRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    CUDAStatus* status
) {
    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* scalePtr = cuda_buffer_device_ptr(scaleRef);
    void* biasPtr = cuda_buffer_device_ptr(biasRef);
    void* outputPtr = cuda_buffer_device_ptr(outputRef);
    void* bufferRefs[] = {&inputPtr, &scalePtr, &biasPtr, &outputPtr};
    void* uintArgs[] = {&channels, &spatial};

    return cuda_normalization_dispatch_rows(
        contextRef,
        "instancenorm",
        elementDType,
        bufferRefs,
        4,
        uintArgs,
        2,
        batch * channels,
        completionToken,
        status
    );
}
