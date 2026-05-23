//go:build xla

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
DispatchStandardUnary executes a standard unary activation on resident tensors.
*/
func (activation *Activation) DispatchStandardUnary(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernel StandardKernel,
) {
	activation.host.StandardUnary(dst, src, format, kernel)
}
