//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (host *ComputeHost) DispatchActivationSteer(
	destination, base, direction unsafe.Pointer,
	coefficient float32,
	count int,
	format dtype.DType,
) {
	_ = destination
	_ = base
	_ = direction
	_ = coefficient
	_ = count
	_ = format

	host.unavailable()
}
