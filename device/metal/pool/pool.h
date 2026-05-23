#ifndef PUTER_DEVICE_METAL_POOL_H
#define PUTER_DEVICE_METAL_POOL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_vision_status_set(MetalStatus* status, int code, const char* message);

int metal_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
