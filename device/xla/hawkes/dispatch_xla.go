//go:build xla

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkes *Hawkes) HawkesKernelMatrix(
	eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType,
) {
	hawkes.host.DispatchHawkesKernelMatrix(eventTimes, output, eventCount, alpha, beta, format)
}

func (hawkes *Hawkes) HawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
) {
	hawkes.host.DispatchHawkesIntensity(
		eventTimes, queryTimes, output,
		eventCount, queryCount,
		mu, alpha, beta, format,
	)
}

func (hawkes *Hawkes) HawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
) {
	hawkes.host.DispatchHawkesLogLikelihood(
		eventTimes, eventCount, totalT, mu, alpha, beta, output, format,
	)
}

func (hawkesProcess *Hawkes) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	hawkesProcess.host.DispatchMarkovBlanketPartition(
		adjacency, internal, output, nodeCount, internalCount, format,
	)
}

func (hawkesProcess *Hawkes) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	hawkesProcess.host.DispatchMarkovMutualInformation(joint, output, xCount, yCount, format)
}
