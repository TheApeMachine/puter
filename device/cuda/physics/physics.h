#ifndef PUTER_DEVICE_CUDA_PHYSICS_H
#define PUTER_DEVICE_CUDA_PHYSICS_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_physics_register_module_source(const char* source);

const char* cuda_physics_module_source(void);

int cuda_physics_kernel_name(
    char* out,
    size_t outBytes,
    const char* operation,
    int elementDType,
    CUDAStatus* status
);

int cuda_physics_prefixed_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    const char* suffix,
    CUDAStatus* status
);

int cuda_physics_launch_kernel(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_physics_launch_kernel_no_track(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
