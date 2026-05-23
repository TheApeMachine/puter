//go:build darwin && cgo

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
) {
	hawkes.host.DispatchHawkesLogLikelihood(eventTimes, eventCount, totalT, mu, alpha, beta, output, format)
}
