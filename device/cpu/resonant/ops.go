package resonant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
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
			unsafe.Slice((*float32)(x), elementCount),
			unsafe.Slice((*float32)(y), elementCount),
			unsafe.Slice((*float32)(vr), elementCount),
			unsafe.Slice((*float32)(vi), elementCount),
			unsafe.Slice((*float32)(diag), diagCount),
			unsafe.Slice((*float32)(xOut), elementCount),
			unsafe.Slice((*float32)(yOut), elementCount),
			unsafe.Slice((*float32)(aOut), elementCount),
			unsafe.Slice((*float32)(bOut), elementCount),
			unsafe.Slice((*float32)(invROut), elementCount),
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
			unsafe.Slice((*float32)(gradXOut), elementCount),
			unsafe.Slice((*float32)(gradYOut), elementCount),
			unsafe.Slice((*float32)(x), elementCount),
			unsafe.Slice((*float32)(y), elementCount),
			unsafe.Slice((*float32)(diag), diagCount),
			unsafe.Slice((*float32)(a), elementCount),
			unsafe.Slice((*float32)(b), elementCount),
			unsafe.Slice((*float32)(invR), elementCount),
			unsafe.Slice((*float32)(gradX), elementCount),
			unsafe.Slice((*float32)(gradY), elementCount),
			unsafe.Slice((*float32)(gradVR), elementCount),
			unsafe.Slice((*float32)(gradVI), elementCount),
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
	if count == 0 {
		return nil
	}

	return unsafe.Slice((*uint16)(pointer), count)
}
