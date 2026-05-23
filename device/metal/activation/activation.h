#ifndef PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_H
#define PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_activation_status_clear(MetalStatus* status);

void metal_activation_status_set(MetalStatus* status, int code, const char* message);

const char* metal_activation_element_dtype_suffix(int elementDType);

int metal_activation_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    MetalStatus* status
);

uint32_t metal_activation_vector_launch_count(uint32_t count, int elementDType);

#ifdef __cplusplus
}
#endif

#endif
