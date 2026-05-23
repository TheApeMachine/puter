#include "avgpool.h"
#include "pool.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_avg_pool2d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_vision_dispatch_pool2d(
        contextRef,
        "avg_pool2d",
        elementDType,
        inputRef,
        outRef,
        batch,
        channels,
        inHeight,
        inWidth,
        outHeight,
        outWidth,
        completionToken,
        status
    );
}
