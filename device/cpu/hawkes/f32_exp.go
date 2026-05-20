package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/activation"
)

/*
hawkesExpScalar applies the platform activation exp kernel to one lane.
*/
func hawkesExpScalar(value float32) float32 {
	scratch := [1]float32{value}
	activation.Exp(
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		len(scratch),
		dtype.Float32,
	)

	return scratch[0]
}
