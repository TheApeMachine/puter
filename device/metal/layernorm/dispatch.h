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

int metal_dispatch_layernorm_stats(
	MetalDeviceRef contextRef,
	MetalBufferRef inputRef,
	MetalBufferRef rowStatsRef,
	uint32_t rows,
	uint32_t cols,
	uint64_t completionToken,
	MetalStatus* status
);

int metal_dispatch_layernorm_apply(
	MetalDeviceRef contextRef,
	MetalBufferRef inputRef,
	MetalBufferRef scaleRef,
	MetalBufferRef biasRef,
	MetalBufferRef outRef,
	MetalBufferRef rowStatsRef,
	uint32_t rows,
	uint32_t cols,
	uint64_t completionToken,
	MetalStatus* status
);

int metal_dispatch_layernorm_rmsnorm(
	MetalDeviceRef contextRef,
	int elementDType,
	MetalBufferRef inputRef,
	MetalBufferRef scaleRef,
	MetalBufferRef outRef,
	uint32_t rows,
	uint32_t cols,
	float epsilon,
	uint64_t completionToken,
	MetalStatus* status
);

int metal_dispatch_layernorm_adaptive_rmsnorm(
	MetalDeviceRef contextRef,
	int elementDType,
	MetalBufferRef inputRef,
	MetalBufferRef modulationRef,
	MetalBufferRef outRef,
	uint32_t rows,
	uint32_t cols,
	uint32_t rowsPerBatch,
	uint32_t modulationCols,
	float epsilon,
	uint64_t completionToken,
	MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
