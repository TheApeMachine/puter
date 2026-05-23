//go:build !darwin || !cgo

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesKernelMatrix(eventTimes, output unsafe.Pointer, eventCount int, alpha, beta float32, format dtype.DType,) {
	hawkes.stubHost()
}
