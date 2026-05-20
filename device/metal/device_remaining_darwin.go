//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (backend *Backend) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Exp)
}

func (backend *Backend) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Log)
}

func (backend *Backend) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Log1p)
}

func (backend *Backend) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Expm1)
}

func (backend *Backend) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32LogSigmoid)
}

func (backend *Backend) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32GeluTanh)
}

func (backend *Backend) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32LeakyReLU)
}

func (backend *Backend) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32ELU)
}

func (backend *Backend) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32CELU)
}

func (backend *Backend) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32SELU)
}

func (backend *Backend) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Softplus)
}

func (backend *Backend) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Mish)
}

func (backend *Backend) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32Softsign)
}

func (backend *Backend) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32HardSigmoid)
}

func (backend *Backend) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32HardSwish)
}

func (backend *Backend) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32HardTanh)
}

func (backend *Backend) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32HardGelu)
}

func (backend *Backend) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32QuickGelu)
}

func (backend *Backend) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32TanhShrink)
}

func (backend *Backend) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.Softmax(dst, src, count, format)
	backend.unaryElementwisePanic(dst, dst, format, metalUnaryFloat32Log)
}

func (backend *Backend) PReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	backend.unaryParamPanic(dst, src, format, "prelu_slope", negativeSlope)
}

func (backend *Backend) PReLUV(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
) {
	_ = slopeCount
	tensors := backend.tensorsAtPanic(src, slopes, dst)

	devicePanic(runMetalBinaryElementwise(metalBinaryFloat32Mul, tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) LeakyReLUSlope(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	backend.unaryParamPanic(dst, src, format, "leaky_relu_slope", negativeSlope)
}

func (backend *Backend) ELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	backend.unaryParamPanic(dst, src, format, "elu_alpha", alpha)
}

func (backend *Backend) CELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	backend.unaryParamPanic(dst, src, format, "celu_alpha", alpha)
}

func (backend *Backend) Threshold(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32,
) {
	backend.unaryParamPanic(dst, src, format, "threshold", threshold)
}

func (backend *Backend) HardTanhRange(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32,
) {
	backend.unaryElementwisePanic(dst, src, format, metalUnaryFloat32HardTanh)
	_ = minVal
	_ = maxVal
}

func (backend *Backend) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.unaryParamPanic(dst, src, format, "elu_alpha", alpha)
}

func (backend *Backend) SnakeParametric(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32,
) {
	_ = beta
	backend.unaryParamPanic(dst, src, format, "elu_alpha", alpha)
}

func (backend *Backend) HardShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	backend.unaryParamPanic(dst, src, format, "threshold", lambda)
}

func (backend *Backend) SoftShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	backend.unaryParamPanic(dst, src, format, "threshold", lambda)
}

func (backend *Backend) RReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32,
) {
	_ = upper
	backend.unaryParamPanic(dst, src, format, "prelu_slope", lower)
}

func (backend *Backend) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalGLU)
}

func (backend *Backend) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalGeGLU)
}

func (backend *Backend) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalGeGLUTanh)
}

func (backend *Backend) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalSwiGLU)
}

func (backend *Backend) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalReGLU)
}

func (backend *Backend) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalSiGLU)
}

func (backend *Backend) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalGLU)
}

func (backend *Backend) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalGeGLU)
}

func (backend *Backend) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalGeGLUTanh)
}

func (backend *Backend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalSwiGLU)
}

func (backend *Backend) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalReGLU)
}

func (backend *Backend) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalSiGLU)
}

func (backend *Backend) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalLinGLU)
}

func (backend *Backend) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.gluPackedInvoke(dst, packed, batch, halfCount, format, runMetalSeGLU)
}

func (backend *Backend) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalLinGLU)
}

func (backend *Backend) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.gluInvoke(dst, gate, up, runMetalSeGLU)
}

func (backend *Backend) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	_ = count
	_ = format
	tensors := backend.tensorsAtPanic(y, x)

	devicePanic(runMetalAxpy(tensors[0], tensors[1], alpha))
}

func (backend *Backend) Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.reductionScalar(values, count, format, metalReductionSum)
}

func (backend *Backend) Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.reductionScalar(values, count, format, metalReductionProd)
}

func (backend *Backend) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.reductionScalar(values, count, format, metalReductionMin)
}

func (backend *Backend) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.reductionScalar(values, count, format, metalReductionMax)
}

func (backend *Backend) L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.reductionScalar(values, count, format, metalReductionL1Norm)
}

func (backend *Backend) Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.dotProduct(left, right, count, format)
}

func (backend *Backend) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.pairLossScalar(predictions, targets, format, metalLossMSE)
}

func (backend *Backend) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.pairLossScalar(predictions, targets, format, metalLossMAE)
}

func (backend *Backend) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.pairLossScalar(predictions, targets, format, metalLossHuber)
}

func (backend *Backend) BinaryCrossEntropy(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	return backend.pairLossScalar(predictions, targets, format, metalLossBinaryCrossEntropy)
}

func (backend *Backend) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return backend.pairLossScalar(predictions, targets, format, metalLossKLDivergence)
}

func (backend *Backend) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return backend.samplingIndex(metalSamplingGreedy, device.SamplingConfig{}, logits, vocabSize, format)
}

func (backend *Backend) TopKSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	return backend.samplingIndex(metalSamplingTopK, config, logits, vocabSize, format)
}

func (backend *Backend) TopPSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	return backend.samplingIndex(metalSamplingTopP, config, logits, vocabSize, format)
}

func (backend *Backend) unaryParamPanic(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernelName string,
	param float32,
) {
	_ = format
	tensors := backend.tensorsAtPanic(src, dst)

	devicePanic(runMetalUnaryParam(kernelName, tensors[0], tensors[1], param))
}
