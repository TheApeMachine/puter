#ifndef PUTER_DEVICE_METAL_ATTENTION_ATTENTION_H
#define PUTER_DEVICE_METAL_ATTENTION_ATTENTION_H

#include "../internal/bridge/bridge_transformer_private.h"

#ifdef __cplusplus
extern "C" {
#endif

void metal_attention_status_clear(MetalStatus* status);

#ifdef __cplusplus
}
#endif

#endif
