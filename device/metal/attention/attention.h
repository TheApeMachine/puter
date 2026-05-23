#ifndef PUTER_DEVICE_METAL_ATTENTION_ATTENTION_H
#define PUTER_DEVICE_METAL_ATTENTION_ATTENTION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_transformer_status_set(MetalStatus* status, int code, const char* message);

int metal_transformer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);

void metal_attention_status_clear(MetalStatus* status);

#ifdef __cplusplus
}
#endif

#endif
