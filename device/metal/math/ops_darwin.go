//go:build darwin && cgo

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
	math.host.DispatchInvSqrtDimScale(out, input, dim, format)
}

func (math *Math) LogSumExp(
	input, output unsafe.Pointer,
	cols int,
	format dtype.DType,
) {
	math.host.DispatchLogSumExp(input, output, cols, format)
}

func (math *Math) Outer(
	left, right, output unsafe.Pointer,
	leftCount, rightCount int,
	format dtype.DType,
) {
	math.host.DispatchOuter(left, right, output, leftCount, rightCount, format)
}
