#ifndef PUTER_DEVICE_METAL_REDUCTION_H
#define PUTER_DEVICE_METAL_REDUCTION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_reduction_status_clear(MetalStatus* status);

void metal_reduction_status_set(MetalStatus* status, int code, const char* message);

int metal_reduction_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
