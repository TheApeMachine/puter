#include "conv2d.h"
#include "convolution.h"
#include "../pool/pool.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_conv2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), "conv2d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * outChannels * outHeight * outWidth;
    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inChannels length:sizeof(inChannels) atIndex:5];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:6];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:7];
            [encoder setBytes:&outChannels length:sizeof(outChannels) atIndex:8];
            [encoder setBytes:&kernelHeight length:sizeof(kernelHeight) atIndex:9];
            [encoder setBytes:&kernelWidth length:sizeof(kernelWidth) atIndex:10];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:11];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:12];
        }
    );
}
