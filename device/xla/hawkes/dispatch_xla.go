//go:build xla

package hawkes

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesKernelMatrix(eventTimes, output unsafe.Pointer, eventCount int, alpha, beta float32, format dtype.DType,) {
	hawkes.unimplemented("HawkesKernelMatrix")
}

func (hawkes *Hawkes) HawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer, eventCount, queryCount int, mu, alpha, beta float32, format dtype.DType,) {
	hawkes.unimplemented("HawkesIntensity")
}

func (hawkes *Hawkes) HawkesLogLikelihood(eventTimes unsafe.Pointer, eventCount int, totalT, mu, alpha, beta float32, output unsafe.Pointer, format dtype.DType,) {
	hawkes.unimplemented("HawkesLogLikelihood")
}

func (hawkesProcess *Hawkes) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	hawkesProcess.unimplemented("MarkovBlanketPartition")
}

func (hawkesProcess *Hawkes) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	hawkesProcess.unimplemented("MarkovMutualInformation")
}

