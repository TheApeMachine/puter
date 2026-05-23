#ifndef PUTER_DEVICE_CUDA_INTERNAL_BRIDGE_CORE_PRIVATE_H
#define PUTER_DEVICE_CUDA_INTERNAL_BRIDGE_CORE_PRIVATE_H

#include "core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUDAContext {
    int deviceIndex;
    void* moduleCache;
    void* moduleLock;
    CUDAStreamRef defaultStream;
    CUDAStreamRef uploadStream;
} CUDAContext;

typedef struct CUDADeferredCompletion {
    uint64_t token;
    CUDAEventRef event;
} CUDADeferredCompletion;

CUDAContext* cuda_context_from_ref(CUDADeviceRef contextRef);

int cuda_context_prepare(
    CUDADeviceRef contextRef,
    CUDAStatus* status,
    CUDAContext** context,
    CUDAStreamRef* stream
);

CUDAKernelRef cuda_get_kernel(
    CUDAContext* context,
    const char* moduleSource,
    const char* kernelName,
    CUDAStatus* status
);

int cuda_launch_1d(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    uint32_t count,
    void** args,
    size_t argBytes,
    CUDAStatus* status
);

void cuda_track_completion(
    CUDAContext* context,
    CUDAStreamRef stream,
    uint64_t completionToken,
    CUDAEventRef event,
    CUDAStatus* status
);

const char* cuda_element_dtype_suffix(int elementDType);

int cuda_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    CUDAStatus* status
);

void* cuda_buffer_device_ptr(CUDABufferRef buffer);

CUDAStreamRef cuda_context_upload_stream(CUDADeviceRef device);

#ifdef __cplusplus
}
#endif

#endif
