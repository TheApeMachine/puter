package hawkes

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultHawkes = New()

func HawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType) {
	defaultHawkes.HawkesIntensity(eventTimes, queryTimes, output, eventCount, queryCount, mu, alpha, beta, format)
}

func HawkesKernelMatrix(eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType) {
	defaultHawkes.HawkesKernelMatrix(eventTimes, output, eventCount, alpha, beta, format)
}

func HawkesLogLikelihood(eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType) {
	defaultHawkes.HawkesLogLikelihood(eventTimes, eventCount, totalT, mu, alpha, beta, output, format)
}

func MarkovBlanketPartition(adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType) {
	defaultHawkes.MarkovBlanketPartition(adjacency, internal, output, nodeCount, internalCount, format)
}

func MarkovMutualInformation(joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType) {
	defaultHawkes.MarkovMutualInformation(joint, output, xCount, yCount, format)
}
