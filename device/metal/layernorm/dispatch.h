#ifndef PUTER_DEVICE_METAL_LAYERNORM_DISPATCH_H
#define PUTER_DEVICE_METAL_LAYERNORM_DISPATCH_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_layernorm(
	MetalDeviceRef contextRef,
	int elementDType,
	MetalBufferRef inputRef,
	MetalBufferRef scaleRef,
	MetalBufferRef biasRef,
	MetalBufferRef outRef,
	uint32_t rows,
	uint32_t cols,
	uint64_t completionToken,
	MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
