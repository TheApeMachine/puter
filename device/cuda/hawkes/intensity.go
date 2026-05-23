//go:build cuda

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
) {
	hawkes.host.DispatchHawkesIntensity(eventTimes, queryTimes, output, eventCount, queryCount, mu, alpha, beta, format)
}
