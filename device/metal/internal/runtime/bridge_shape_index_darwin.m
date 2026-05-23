#include "bridge_shape_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static int metal_shape_named_dispatch(
    MetalDeviceRef contextRef,
    int elementDType,
    const char* operationName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalShapeValidatedEncodeBlock encode
) {
    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch_validated(
        contextRef, kernelName, threadCount, completionToken, status, encode
    );
}

static int metal_shape_named_dispatch_plain(
    MetalDeviceRef contextRef,
    int elementDType,
    const char* operationName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalShapeEncodeBlock encode
) {
    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef, kernelName, threadCount, completionToken, status, encode
    );
}

int metal_dispatch_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef sourceRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t sourceRows,
    uint32_t inner,
    uint32_t outRows,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (sourceRef == NULL || indicesRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch(
        contextRef, elementDType, "gather", (NSUInteger)outRows * inner,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sourceRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBuffer:validationBuffer offset:0 atIndex:3];
            [encoder setBytes:&sourceRows length:sizeof(sourceRows) atIndex:4];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:5];
            [encoder setBytes:&outRows length:sizeof(outRows) atIndex:6];
        }
    );
}

int metal_dispatch_scatter(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef targetRef,
    MetalBufferRef indicesRef,
    MetalBufferRef updatesRef,
    MetalBufferRef outRef,
    uint32_t targetRows,
    uint32_t inner,
    uint32_t updateRows,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (targetRef == NULL || indicesRef == NULL || updatesRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch(
        contextRef, elementDType, "scatter", (NSUInteger)targetRows * inner,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)targetRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)updatesRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBuffer:validationBuffer offset:0 atIndex:4];
            [encoder setBytes:&targetRows length:sizeof(targetRows) atIndex:5];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:6];
            [encoder setBytes:&updateRows length:sizeof(updateRows) atIndex:7];
        }
    );
}

int metal_dispatch_page_write(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef storageRef,
    MetalBufferRef valuesRef,
    MetalBufferRef pageIDsRef,
    MetalBufferRef offsetsRef,
    MetalBufferRef outRef,
    uint32_t pageCount,
    uint32_t pageSize,
    uint32_t inner,
    uint32_t valueRows,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (storageRef == NULL || valuesRef == NULL || pageIDsRef == NULL || offsetsRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch(
        contextRef, elementDType, "page_write", (NSUInteger)pageCount * pageSize * inner,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)storageRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)valuesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)pageIDsRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)offsetsRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBuffer:validationBuffer offset:0 atIndex:5];
            [encoder setBytes:&pageCount length:sizeof(pageCount) atIndex:6];
            [encoder setBytes:&pageSize length:sizeof(pageSize) atIndex:7];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:8];
            [encoder setBytes:&valueRows length:sizeof(valueRows) atIndex:9];
        }
    );
}

int metal_dispatch_page_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef storageRef,
    MetalBufferRef pageTableRef,
    MetalBufferRef outRef,
    uint32_t pageCount,
    uint32_t pageSize,
    uint32_t inner,
    uint32_t outRows,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (storageRef == NULL || pageTableRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch(
        contextRef, elementDType, "page_gather", (NSUInteger)outRows * inner,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)storageRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)pageTableRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBuffer:validationBuffer offset:0 atIndex:3];
            [encoder setBytes:&pageCount length:sizeof(pageCount) atIndex:4];
            [encoder setBytes:&pageSize length:sizeof(pageSize) atIndex:5];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:6];
            [encoder setBytes:&outRows length:sizeof(outRows) atIndex:7];
        }
    );
}

int metal_dispatch_where(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef maskRef,
    MetalBufferRef positiveRef,
    MetalBufferRef negativeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (maskRef == NULL || positiveRef == NULL || negativeRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch_plain(
        contextRef, elementDType, "where", (NSUInteger)((count + 3u) / 4u),
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)maskRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)positiveRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)negativeRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
        }
    );
}

int metal_dispatch_masked_fill(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef scalarRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || maskRef == NULL || scalarRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_named_dispatch_plain(
        contextRef, elementDType, "masked_fill", (NSUInteger)((count + 3u) / 4u),
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)maskRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)scalarRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
        }
    );
}

int metal_dispatch_transpose(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint32_t* permutation,
    const uint32_t* inputStrides,
    const uint32_t* outStrides,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL ||
        permutation == NULL || inputStrides == NULL || outStrides == NULL) {
        metal_shape_status_set(status, -2, "nil Metal transpose argument");
        return -2;
    }

    return metal_shape_named_dispatch(
        contextRef, elementDType, "transpose", (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBuffer:validationBuffer offset:0 atIndex:2];
            [encoder setBytes:&rank length:sizeof(rank) atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
            [encoder setBytes:permutation length:sizeof(uint32_t) * rank atIndex:5];
            [encoder setBytes:inputStrides length:sizeof(uint32_t) * rank atIndex:6];
            [encoder setBytes:outStrides length:sizeof(uint32_t) * rank atIndex:7];
        }
    );
}
