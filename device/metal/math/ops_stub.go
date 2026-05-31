//go:build !darwin || !cgo

package math

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (math *Math) InvSqrtDimScale(
	out, input unsafe.Pointer,
	dim int32,
	format dtype.DType,
) {
	math.host.NeedsPlatform()
}

func (math *Math) LogSumExp(
	input, output unsafe.Pointer,
	cols int,
	format dtype.DType,
) {
	math.host.NeedsPlatform()
}

func (math *Math) Outer(
	left, right, output unsafe.Pointer,
	leftCount, rightCount int,
	format dtype.DType,
) {
	math.host.NeedsPlatform()
}
