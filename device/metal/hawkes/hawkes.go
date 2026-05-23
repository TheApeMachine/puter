package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type Hawkes struct {
	host Host
}

func New(host Host) Hawkes {
	return Hawkes{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchHawkesIntensity(
		eventTimes, queryTimes, output unsafe.Pointer,
		eventCount, queryCount int,
		mu, alpha, beta float32,
		format dtype.DType,
	)
	DispatchHawkesKernelMatrix(
		eventTimes, output unsafe.Pointer,
		eventCount int,
		alpha, beta float32,
		format dtype.DType,
	)
	DispatchHawkesLogLikelihood(
		eventTimes unsafe.Pointer,
		eventCount int,
		totalT, mu, alpha, beta float32,
		output unsafe.Pointer,
		format dtype.DType,
	)
}
