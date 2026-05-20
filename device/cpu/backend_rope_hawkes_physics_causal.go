package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/cpu/hawkes"
	"github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/cpu/rope"
)

func (backend *Backend) RoPE(
	config device.RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	rope.RoPE(ropeConfig(config), input, output, seqLen, numHeads, headDim, format)
}

func (backend *Backend) RoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType,
) {
	rope.RoPEPairs(output, input, cosBuffer, sinBuffer, halfDim, format)
}

func (backend *Backend) HawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
) {
	hawkes.HawkesIntensity(eventTimes, queryTimes, output, eventCount, queryCount, mu, alpha, beta, format)
}

func (backend *Backend) HawkesKernelMatrix(
	eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType,
) {
	hawkes.HawkesKernelMatrix(eventTimes, output, eventCount, alpha, beta, format)
}

func (backend *Backend) HawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
) {
	hawkes.HawkesLogLikelihood(eventTimes, eventCount, totalT, mu, alpha, beta, output, format)
}

func (backend *Backend) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	hawkes.MarkovMutualInformation(joint, output, xCount, yCount, format)
}

func (backend *Backend) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	hawkes.MarkovBlanketPartition(adjacency, internal, output, nodeCount, internalCount, format)
}

func (backend *Backend) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	physics.Laplacian(input, output, dims, spacing, format)
}

func (backend *Backend) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.Laplacian4(input, output, count, spacing, format)
}

func (backend *Backend) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.Grad1D(input, output, count, spacing, format)
}

func (backend *Backend) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.Divergence1D(input, output, count, spacing, format)
}

func (backend *Backend) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.FFT1D(realIn, imagIn, realOut, imagOut, count, format)
}

func (backend *Backend) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.IFFT1D(realIn, imagIn, realOut, imagOut, count, format)
}

func (backend *Backend) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.QuantumPotential(density, output, count, spacing, format)
}

func (backend *Backend) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.BohmianVelocity(phase, output, count, spacing, format)
}

func (backend *Backend) MadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	physics.MadelungContinuity(density, velocity, residual, count, spacing, format)
}

func (backend *Backend) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	causal.Cholesky(input, output, matrixOrder, format)
}

func (backend *Backend) BackdoorAdjustment(
	conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType,
) {
	causal.BackdoorAdjustment(conditional, marginalZ, output, xCount, zCount, yCount, format)
}

func (backend *Backend) FrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType,
) {
	causal.FrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output, xCount, mediatorCount, yCount, format)
}

func (backend *Backend) DoIntervene(
	adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType,
) {
	causal.DoIntervene(adjacency, intervened, output, nodeCount, intervenedCount, format)
}

func (backend *Backend) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	causal.CATE(treated, control, output, count, format)
}

func (backend *Backend) Counterfactual(
	observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType,
) {
	causal.Counterfactual(observedY, observedX, counterfactualX, output, count, slope, format)
}

func (backend *Backend) IVEstimate(
	instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	causal.IVEstimate(instrument, treatment, outcome, count, output, format)
}

func (backend *Backend) DAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	causal.DAGMarkovFactorization(conditionals, conditionalCount, output, format)
}

func (backend *Backend) MarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	causal.MarkovFlowActive(mutualInformation, partition, output, nodeCount, format)
}

func (backend *Backend) MarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	causal.MarkovFlowInternal(mutualInformation, partition, output, nodeCount, format)
}
