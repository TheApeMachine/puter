#include "product.h"
#include "matmul.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_matmul(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_matmul_dispatch_tiled(
        contextRef,
        "matmul",
        elementDType,
        leftRef,
        rightRef,
        NULL,
        outRef,
        rows,
        inner,
        cols,
        0,
        completionToken,
        status
    );
}

int cuda_dispatch_matmul_add(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
) {
    return cuda_matmul_dispatch_tiled(
        contextRef,
        "matmul_add",
        elementDType,
        leftRef,
        rightRef,
        biasRef,
        outRef,
        rows,
        inner,
        cols,
        1,
        completionToken,
        status
    );
}
