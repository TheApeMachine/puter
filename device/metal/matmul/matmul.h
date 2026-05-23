#ifndef PUTER_DEVICE_METAL_MATMUL_H
#define PUTER_DEVICE_METAL_MATMUL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_matmul_status_clear(MetalStatus* status);

void metal_matmul_status_set(MetalStatus* status, int code, const char* message);

int metal_matmul_kernel_name(
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
