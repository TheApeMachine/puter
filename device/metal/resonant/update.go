//go:build darwin && cgo

package resonant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (resonant Resonant) ResonantUpdateForward(
	x, y, vr, vi, diag unsafe.Pointer,
	xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	resonant.host.DispatchResonantUpdateForward(
		x, y, vr, vi, diag,
		xOut, yOut, aOut, bOut, invROut,
		batchTime, headCount, headDim,
		config,
		format,
	)
}

func (resonant Resonant) ResonantUpdateBackward(
	gradXOut, gradYOut unsafe.Pointer,
	x, y, diag, a, b, invR unsafe.Pointer,
	gradX, gradY, gradVR, gradVI unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	resonant.host.DispatchResonantUpdateBackward(
		gradXOut, gradYOut,
		x, y, diag, a, b, invR,
		gradX, gradY, gradVR, gradVI,
		batchTime, headCount, headDim,
		config,
		format,
	)
}
