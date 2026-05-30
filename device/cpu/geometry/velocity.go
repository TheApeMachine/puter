package geometry

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/elementwise"
)

func runPhaseVelocityFloat32(
	destination, current, previous unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	elementwise.SubFloat32Native(
		unsafe.Slice((*float32)(destination), count),
		unsafe.Slice((*float32)(current), count),
		unsafe.Slice((*float32)(previous), count),
	)
}

func runPhaseVelocityFloat16(
	destination, current, previous unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	elementwise.SubFloat16Native(
		unsafe.Slice((*dtype.F16)(destination), count),
		unsafe.Slice((*dtype.F16)(current), count),
		unsafe.Slice((*dtype.F16)(previous), count),
	)
}

func runPhaseVelocityBFloat16(
	destination, current, previous unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	elementwise.SubBFloat16Native(
		unsafe.Slice((*dtype.BF16)(destination), count),
		unsafe.Slice((*dtype.BF16)(current), count),
		unsafe.Slice((*dtype.BF16)(previous), count),
	)
}
