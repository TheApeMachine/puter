#include "gated.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

#include <stdint.h>
#include <stdio.h>

static int cuda_gated_launch_tensor(
    CUDADeviceRef contextRef,
    const char* kernelName,
    int elementDType,
    CUDABufferRef destinationRef,
    CUDABufferRef gateRef,
    CUDABufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_activation_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (destinationRef == NULL || gateRef == NULL || upRef == NULL || kernelName == NULL) {
        cuda_activation_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* moduleSource = cuda_activation_module_source();

    if (moduleSource == NULL) {
        cuda_activation_status_set(status, -7, "CUDA activation module source not registered");
        return -7;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* destinationPtr = cuda_buffer_device_ptr(destinationRef);
    void* gatePtr = cuda_buffer_device_ptr(gateRef);
    void* upPtr = cuda_buffer_device_ptr(upRef);
    void* args[] = {&destinationPtr, &gatePtr, &upPtr, &count};
    uint32_t launchCount = cuda_activation_vector_launch_count(count, elementDType);
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

static int cuda_gated_launch_packed(
    CUDADeviceRef contextRef,
    const char* kernelName,
    CUDABufferRef destinationRef,
    CUDABufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_activation_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (destinationRef == NULL || packedRef == NULL || kernelName == NULL) {
        cuda_activation_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* moduleSource = cuda_activation_module_source();

    if (moduleSource == NULL) {
        cuda_activation_status_set(status, -7, "CUDA activation module source not registered");
        return -7;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* destinationPtr = cuda_buffer_device_ptr(destinationRef);
    void* packedPtr = cuda_buffer_device_ptr(packedRef);
    void* args[] = {&destinationPtr, &packedPtr, &inner, &count};
    int launchCode = cuda_launch_1d(context, kernel, stream, count, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

static const char* cuda_gated_tensor_kernel_name(const char* prefix, int elementDType, CUDAStatus* status) {
    static __thread char kernelName[128];
    const char* suffix = cuda_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA gated dtype");
        return NULL;
    }

    if (cuda_activation_compose_kernel_name(kernelName, sizeof(kernelName), prefix, suffix, status) != 0) {
        return NULL;
    }

    return kernelName;
}

static const char* cuda_gated_packed_kernel_name(const char* prefix, int elementDType, CUDAStatus* status) {
    static __thread char kernelName[128];
    const char* suffix = cuda_activation_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_activation_status_set(status, -6, "unknown CUDA packed gated dtype");
        return NULL;
    }

    char prefixBuffer[64];
    snprintf(prefixBuffer, sizeof(prefixBuffer), "%s_packed", prefix);

    if (cuda_activation_compose_kernel_name(kernelName, sizeof(kernelName), prefixBuffer, suffix, status) != 0) {
        return NULL;
    }

    return kernelName;
}

#define CUDA_DISPATCH_GATED_TENSOR(name) \
int cuda_dispatch_##name( \
    CUDADeviceRef contextRef, \
    int elementDType, \
    CUDABufferRef destinationRef, \
    CUDABufferRef gateRef, \
    CUDABufferRef upRef, \
    uint32_t count, \
    uint64_t completionToken, \
    CUDAStatus* status \
) { \
    const char* kernelName = cuda_gated_tensor_kernel_name(#name, elementDType, status); \
    if (kernelName == NULL) { \
        return status != NULL && status->code != 0 ? status->code : -6; \
    } \
    return cuda_gated_launch_tensor( \
        contextRef, \
        kernelName, \
        elementDType, \
        destinationRef, \
        gateRef, \
        upRef, \
        count, \
        completionToken, \
        status \
    ); \
}

#define CUDA_DISPATCH_GATED_PACKED(name) \
int cuda_dispatch_##name##_packed( \
    CUDADeviceRef contextRef, \
    int elementDType, \
    CUDABufferRef destinationRef, \
    CUDABufferRef packedRef, \
    uint32_t inner, \
    uint32_t count, \
    uint64_t completionToken, \
    CUDAStatus* status \
) { \
    const char* kernelName = cuda_gated_packed_kernel_name(#name, elementDType, status); \
    if (kernelName == NULL) { \
        return status != NULL && status->code != 0 ? status->code : -6; \
    } \
    return cuda_gated_launch_packed( \
        contextRef, \
        kernelName, \
        destinationRef, \
        packedRef, \
        inner, \
        count, \
        completionToken, \
        status \
    ); \
}

CUDA_DISPATCH_GATED_TENSOR(swiglu)
CUDA_DISPATCH_GATED_PACKED(swiglu)
CUDA_DISPATCH_GATED_TENSOR(geglu)
CUDA_DISPATCH_GATED_PACKED(geglu)
CUDA_DISPATCH_GATED_TENSOR(glu)
CUDA_DISPATCH_GATED_PACKED(glu)
CUDA_DISPATCH_GATED_TENSOR(reglu)
CUDA_DISPATCH_GATED_PACKED(reglu)
CUDA_DISPATCH_GATED_TENSOR(siglu)
CUDA_DISPATCH_GATED_PACKED(siglu)
CUDA_DISPATCH_GATED_TENSOR(seglu)
CUDA_DISPATCH_GATED_PACKED(seglu)
CUDA_DISPATCH_GATED_TENSOR(linglu)
CUDA_DISPATCH_GATED_PACKED(linglu)
CUDA_DISPATCH_GATED_TENSOR(geglu_tanh)
CUDA_DISPATCH_GATED_PACKED(geglu_tanh)
