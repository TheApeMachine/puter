#include "product.h"
#include "matmul.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_matmul(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        metal_matmul_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    int mpsCode = -100;

    if (elementDType != MetalElementDTypeBFloat16) {
        mpsCode = metal_matmul_dispatch_mps(
            contextRef,
            elementDType,
            leftRef,
            rightRef,
            outRef,
            rows,
            inner,
            cols,
            completionToken,
            status
        );
    }

    if (mpsCode != -100) {
        return mpsCode;
    }

    char kernelName[128];
    int nameCode = metal_matmul_kernel_name(
        kernelName,
        sizeof(kernelName),
        "matmul",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_matmul_dispatch(
        contextRef,
        kernelName,
        rows,
        cols,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:4];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:5];
        }
    );
}

int metal_dispatch_matmul_add(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (leftRef == NULL || rightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_matmul_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_matmul_kernel_name(
        kernelName,
        sizeof(kernelName),
        "matmul_add",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_matmul_dispatch(
        contextRef,
        kernelName,
        rows,
        cols,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:5];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        }
    );
}
