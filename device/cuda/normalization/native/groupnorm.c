#include "groupnorm.h"
#include "normalization.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_groupnorm(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef outputRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    CUDAStatus* status
) {
    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* scalePtr = cuda_buffer_device_ptr(scaleRef);
    void* biasPtr = cuda_buffer_device_ptr(biasRef);
    void* outputPtr = cuda_buffer_device_ptr(outputRef);
    void* bufferRefs[] = {&inputPtr, &scalePtr, &biasPtr, &outputPtr};
    void* uintArgs[] = {&channels, &spatial, &groups};

    return cuda_normalization_dispatch_rows(
        contextRef,
        "groupnorm",
        elementDType,
        bufferRefs,
        4,
        uintArgs,
        3,
        batch * groups,
        completionToken,
        status
    );
}
