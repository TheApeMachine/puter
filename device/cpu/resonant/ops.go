package resonant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func requireResonantDType(format dtype.DType) {
	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		return
	default:
		panic("resonant: unsupported dtype")
	}
}

func resonantElementCount(batchTime, headCount, headDim int) int {
	return batchTime * headCount * headDim
}

func (resonant Resonant) ResonantUpdateForward(
	x, y, vr, vi, diag unsafe.Pointer,
	xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	requireResonantDType(format)

	elementCount := resonantElementCount(batchTime, headCount, headDim)

	if elementCount == 0 {
		return
	}

	diagCount := headCount * headDim

	switch format {
	case dtype.Float32:
		ResonantUpdateForwardGeneric(
			dispatch.Float32Slice(x, elementCount),
			dispatch.Float32Slice(y, elementCount),
			dispatch.Float32Slice(vr, elementCount),
			dispatch.Float32Slice(vi, elementCount),
			dispatch.Float32Slice(diag, diagCount),
			dispatch.Float32Slice(xOut, elementCount),
			dispatch.Float32Slice(yOut, elementCount),
			dispatch.Float32Slice(aOut, elementCount),
			dispatch.Float32Slice(bOut, elementCount),
			dispatch.Float32Slice(invROut, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.Float16:
		ResonantUpdateForwardFloat16(
			uint16Slice(x, elementCount),
			uint16Slice(y, elementCount),
			uint16Slice(vr, elementCount),
			uint16Slice(vi, elementCount),
			uint16Slice(diag, diagCount),
			uint16Slice(xOut, elementCount),
			uint16Slice(yOut, elementCount),
			uint16Slice(aOut, elementCount),
			uint16Slice(bOut, elementCount),
			uint16Slice(invROut, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.BFloat16:
		ResonantUpdateForwardBFloat16(
			uint16Slice(x, elementCount),
			uint16Slice(y, elementCount),
			uint16Slice(vr, elementCount),
			uint16Slice(vi, elementCount),
			uint16Slice(diag, diagCount),
			uint16Slice(xOut, elementCount),
			uint16Slice(yOut, elementCount),
			uint16Slice(aOut, elementCount),
			uint16Slice(bOut, elementCount),
			uint16Slice(invROut, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	}
}

func (resonant Resonant) ResonantUpdateBackward(
	gradXOut, gradYOut unsafe.Pointer,
	x, y, diag, a, b, invR unsafe.Pointer,
	gradX, gradY, gradVR, gradVI unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	requireResonantDType(format)

	elementCount := resonantElementCount(batchTime, headCount, headDim)

	if elementCount == 0 {
		return
	}

	diagCount := headCount * headDim

	switch format {
	case dtype.Float32:
		ResonantUpdateBackwardGeneric(
			dispatch.Float32Slice(gradXOut, elementCount),
			dispatch.Float32Slice(gradYOut, elementCount),
			dispatch.Float32Slice(x, elementCount),
			dispatch.Float32Slice(y, elementCount),
			dispatch.Float32Slice(diag, diagCount),
			dispatch.Float32Slice(a, elementCount),
			dispatch.Float32Slice(b, elementCount),
			dispatch.Float32Slice(invR, elementCount),
			dispatch.Float32Slice(gradX, elementCount),
			dispatch.Float32Slice(gradY, elementCount),
			dispatch.Float32Slice(gradVR, elementCount),
			dispatch.Float32Slice(gradVI, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.Float16:
		ResonantUpdateBackwardFloat16(
			uint16Slice(gradXOut, elementCount),
			uint16Slice(gradYOut, elementCount),
			uint16Slice(x, elementCount),
			uint16Slice(y, elementCount),
			uint16Slice(diag, diagCount),
			uint16Slice(a, elementCount),
			uint16Slice(b, elementCount),
			uint16Slice(invR, elementCount),
			uint16Slice(gradX, elementCount),
			uint16Slice(gradY, elementCount),
			uint16Slice(gradVR, elementCount),
			uint16Slice(gradVI, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.BFloat16:
		ResonantUpdateBackwardBFloat16(
			uint16Slice(gradXOut, elementCount),
			uint16Slice(gradYOut, elementCount),
			uint16Slice(x, elementCount),
			uint16Slice(y, elementCount),
			uint16Slice(diag, diagCount),
			uint16Slice(a, elementCount),
			uint16Slice(b, elementCount),
			uint16Slice(invR, elementCount),
			uint16Slice(gradX, elementCount),
			uint16Slice(gradY, elementCount),
			uint16Slice(gradVR, elementCount),
			uint16Slice(gradVI, elementCount),
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	}
}

func uint16Slice(pointer unsafe.Pointer, count int) []uint16 {
	return dispatch.Uint16Slice(pointer, count)
}
