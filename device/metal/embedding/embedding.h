#ifndef PUTER_DEVICE_METAL_EMBEDDING_H
#define PUTER_DEVICE_METAL_EMBEDDING_H

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

#ifdef __cplusplus
}
#endif

#endif
