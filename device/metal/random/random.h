#ifndef PUTER_DEVICE_METAL_RANDOM_H
#define PUTER_DEVICE_METAL_RANDOM_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
metal_random_status_clear resets a MetalStatus block.
*/
void metal_random_status_clear(MetalStatus* status);

/*
metal_random_status_set populates a MetalStatus with a code and message.
*/
void metal_random_status_set(MetalStatus* status, int code, const char* message);

#ifdef __cplusplus
}
#endif

#endif
