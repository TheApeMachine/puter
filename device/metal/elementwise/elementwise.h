#ifndef PUTER_DEVICE_METAL_ELEMENTWISE_H
#define PUTER_DEVICE_METAL_ELEMENTWISE_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_elementwise_status_clear(MetalStatus* status);

void metal_elementwise_status_set(MetalStatus* status, int code, const char* message);

const char* metal_elementwise_element_dtype_suffix(int elementDType);

int metal_elementwise_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* dtypeSuffix,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
