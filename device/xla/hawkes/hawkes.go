package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Hawkes implements device.Hawkes for the XLA backend.
*/
type Hawkes struct {
	host Host
}

/*
Host is the XLA dispatch surface hawkes operations call into.
*/
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
	DispatchMarkovMutualInformation(
		joint, output unsafe.Pointer,
		xCount, yCount int,
		format dtype.DType,
	)
	DispatchMarkovBlanketPartition(
		adjacency, internal, output unsafe.Pointer,
		nodeCount, internalCount int,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Hawkes receiver to its XLA dispatch host.
*/
func New(host Host) Hawkes {
	return Hawkes{host: host}
}

func (receiver *Hawkes) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Hawkes) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
