#include "timestep.h"
#include "embedding.h"
#include "../internal/bridge/core_private.h"

int metal_dispatch_timestep_embedding(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef timestepsRef,
    MetalBufferRef outRef,
    float maxPeriod,
    float downscaleFreqShift,
    float timestepDivisor,
    int flipSinToCos,
    uint32_t count,
    uint32_t dim,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (timestepsRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal timestep buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "timestep_embedding", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)count * dim, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)timestepsRef offset:0 atIndex:0];
            [encoder setBytes:&maxPeriod length:sizeof(maxPeriod) atIndex:1];
            [encoder setBytes:&downscaleFreqShift length:sizeof(downscaleFreqShift) atIndex:2];
            [encoder setBytes:&flipSinToCos length:sizeof(flipSinToCos) atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
            [encoder setBytes:&dim length:sizeof(dim) atIndex:6];
            [encoder setBytes:&timestepDivisor length:sizeof(timestepDivisor) atIndex:7];
        }
    );
}
