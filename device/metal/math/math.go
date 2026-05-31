package math

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Math implements device.Math for the Metal backend.
*/
type Math struct {
	host Host
}

/*
New wires a Math receiver to its Metal dispatch host.
*/
func New(host Host) Math {
	return Math{host: host}
}

/*
Host is the Metal dispatch surface math operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchInvSqrtDimScale(out, input unsafe.Pointer, dim int32, format dtype.DType)
	DispatchLogSumExp(input, output unsafe.Pointer, cols int, format dtype.DType)
	DispatchOuter(left, right, output unsafe.Pointer, leftCount, rightCount int, format dtype.DType)
}
