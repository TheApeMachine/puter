//go:build darwin && cgo

package metal

/*
#include "internal/bridge/core.h"
#include <string.h>
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

func partialReductionCount(count uint32) uint32 {
	return (count + 255) / 256
}

func reductionScratchBytes(count uint32) int64 {
	return int64MulChecked(int64(partialReductionCount(count)), 4, "reductionScratchBytes", count)
}

func dotScratchBytes(count uint32) int64 {
	return int64MulChecked(int64(partialReductionCount(count)), 8, "dotScratchBytes", count)
}

func crossEntropyScratchBytes(batch uint32) int64 {
	return int64MulChecked(int64(batch), 4, "crossEntropyScratchBytes", batch)
}

func samplingScoresScratchBytes(paddedCount uint32) int64 {
	return int64MulChecked(int64(paddedCount), 4, "samplingScoresScratchBytes", paddedCount)
}

func samplingIndicesScratchBytes(paddedCount uint32) int64 {
	return int64MulChecked(int64(paddedCount), 4, "samplingIndicesScratchBytes", paddedCount)
}

func (bridge *metalBridge) borrowScratch(byteCount int64) C.MetalBufferRef {
	if byteCount <= 0 {
		return nil
	}

	bufferRef := C.metal_buffer_new_shared(bridge.device, C.longlong(byteCount))

	if bufferRef == nil {
		panic("metal: scratch allocation failed")
	}

	return bufferRef
}

func (bridge *metalBridge) releaseScratch(bufferRef C.MetalBufferRef) {
	if bufferRef == nil {
		return
	}

	C.metal_buffer_release(bufferRef)
}

func (bridge *metalBridge) readFloat32Scalar(bufferRef C.MetalBufferRef) float32 {
	if bufferRef == nil {
		return 0
	}

	bridge.waitIdle()

	contents := C.metal_buffer_contents(bufferRef)

	if contents == nil {
		panic("metal: nil scratch buffer contents")
	}

	return *(*float32)(contents)
}

func (bridge *metalBridge) readInt32Scalar(bufferRef C.MetalBufferRef) int32 {
	if bufferRef == nil {
		return 0
	}

	bridge.waitIdle()

	contents := C.metal_buffer_contents(bufferRef)

	if contents == nil {
		panic("metal: nil scratch buffer contents")
	}

	return *(*int32)(contents)
}

func bufferRefFromPointer(pointer unsafe.Pointer) C.MetalBufferRef {
	return C.MetalBufferRef(pointer)
}

func deviceRefFromContext(contextRef uintptr) C.MetalDeviceRef {
	return C.MetalDeviceRef(unsafe.Pointer(contextRef))
}

func partialCountOrZero(count uint32) uint32 {
	if count == 0 {
		return 0
	}

	return partialReductionCount(count)
}

func int64MulChecked(left, right int64, operation string, input uint32) int64 {
	product := left * right

	if right != 0 && product/right != left {
		panic(fmt.Sprintf(
			"metal: scratch byte count overflow in %s: %d * %d (input=%d)",
			operation,
			left,
			right,
			input,
		))
	}

	if product > math.MaxInt32 {
		panic(fmt.Sprintf(
			"metal: scratch byte count overflow in %s: %d * %d exceeds MaxInt32 (input=%d)",
			operation,
			left,
			right,
			input,
		))
	}

	return product
}
