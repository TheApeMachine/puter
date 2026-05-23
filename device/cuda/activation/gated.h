#ifndef PUTER_DEVICE_CUDA_ACTIVATION_GATED_H
#define PUTER_DEVICE_CUDA_ACTIVATION_GATED_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_swiglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_swiglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_geglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_geglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_glu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_reglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_reglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_siglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_siglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_seglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_seglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_linglu(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_linglu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_geglu_tanh(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_geglu_tanh_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_glu_packed(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
