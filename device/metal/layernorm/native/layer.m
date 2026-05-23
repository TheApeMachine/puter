#include "layer.h"
#include "layernorm.h"
#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

static void metal_layernorm_status_clear(MetalStatus* status) {
	if (status == NULL) {
		return;
	}

	status->code = 0;
	status->message[0] = '\0';
}

static void metal_layernorm_status_set(MetalStatus* status, int code, const char* message) {
	if (status == NULL) {
		return;
	}

	status->code = code;

	if (message == NULL) {
		status->message[0] = '\0';
		return;
	}

	snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

static const char* metal_layernorm_dtype_suffix(int elementDType) {
	switch (elementDType) {
	case MetalElementDTypeFloat32:
		return "float32";
	case MetalElementDTypeFloat16:
		return "float16";
	case MetalElementDTypeBFloat16:
		return "bfloat16";
	default:
		return NULL;
	}
}

static int metal_layernorm_kernel_name(
	char* out,
	size_t outBytes,
	const char* operationName,
	int elementDType,
	MetalStatus* status
) {
	const char* suffix = metal_layernorm_dtype_suffix(elementDType);

	if (operationName == NULL || suffix == NULL) {
		metal_layernorm_status_set(status, -6, "unknown Metal layernorm kernel");
		return -6;
	}

	int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

	if (written <= 0 || (size_t)written >= outBytes) {
		metal_layernorm_status_set(status, -6, "Metal layernorm kernel name overflow");
		return -6;
	}

	return 0;
}

static int metal_layernorm_dispatch(
	MetalDeviceRef contextRef,
	const char* kernelName,
	uint32_t rows,
	uint64_t completionToken,
	MetalStatus* status,
	void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
	@autoreleasepool {
		metal_layernorm_status_clear(status);

		MetalContext* context = (MetalContext*)contextRef;

		if (context == NULL || context->queue == NULL) {
			metal_layernorm_status_set(status, -1, "invalid Metal context");
			return -1;
		}

		id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

		if (pipeline == nil) {
			return status != NULL && status->code != 0 ? status->code : -7;
		}

		id<MTLCommandBuffer> commandBuffer;
		id<MTLComputeCommandEncoder> encoder = metal_get_encoder((MetalContext*)contextRef, &commandBuffer);

		[encoder setComputePipelineState:pipeline];
		encode(encoder);
		[encoder
			dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
			threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
		];
		metal_track_command_completion((MetalContext*)contextRef, commandBuffer, completionToken, NULL);
		metal_end_encoder((MetalContext*)contextRef, encoder, commandBuffer);

		return 0;
	}
}

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
) {
	if (inputRef == NULL || scaleRef == NULL || biasRef == NULL || outRef == NULL) {
		metal_layernorm_status_set(status, -2, "nil Metal buffer");
		return -2;
	}

	char kernelName[128];
	int nameCode = metal_layernorm_kernel_name(
		kernelName,
		sizeof(kernelName),
		"layernorm",
		elementDType,
		status
	);

	if (nameCode != 0) {
		return nameCode;
	}

	return metal_layernorm_dispatch(
		contextRef,
		kernelName,
		rows,
		completionToken,
		status,
		^(id<MTLComputeCommandEncoder> encoder) {
			[encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
			[encoder setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:1];
			[encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
			[encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
			[encoder setBytes:&cols length:sizeof(cols) atIndex:4];
		}
	);
}
