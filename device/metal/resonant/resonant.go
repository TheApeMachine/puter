package resonant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Resonant implements device.Resonant for the Metal backend.
*/
type Resonant struct {
	host Host
}

func New(host Host) Resonant {
	return Resonant{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchResonantUpdateForward(
		x, y, vr, vi, diag unsafe.Pointer,
		xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
		batchTime, headCount, headDim int,
		config device.ResonantUpdateConfig,
		format dtype.DType,
	)
	DispatchResonantUpdateBackward(
		gradXOut, gradYOut unsafe.Pointer,
		x, y, diag, a, b, invR unsafe.Pointer,
		gradX, gradY, gradVR, gradVI unsafe.Pointer,
		batchTime, headCount, headDim int,
		config device.ResonantUpdateConfig,
		format dtype.DType,
	)
}
