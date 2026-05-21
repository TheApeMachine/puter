//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

/*
Resident returns nil when the Metal bridge is unavailable.
*/
func Resident(value tensor.Tensor) unsafe.Pointer {
	_ = value

	return nil
}
