//go:build !darwin || !cgo

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer, eventCount, queryCount int, mu, alpha, beta float32, format dtype.DType,) {
	hawkes.stubHost()
}
