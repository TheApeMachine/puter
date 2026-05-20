//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/pospop"
)

var _ device.Backend = (*Backend)(nil)

func (backend *Backend) CountString(counts *[8]int, str string) {
	pospop.CountString(counts, str)
}
func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	pospop.Count8(counts, buf)
}
func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	pospop.Count16(counts, buf)
}
func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	pospop.Count32(counts, buf)
}
func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	pospop.Count64(counts, buf)
}
func (backend *Backend) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Exp")
}
func (backend *Backend) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Log")
}
func (backend *Backend) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Log1p")
}
func (backend *Backend) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Expm1")
}
func (backend *Backend) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Sigmoid")
}
func (backend *Backend) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "LogSigmoid")
}
func (backend *Backend) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Tanh")
}
func (backend *Backend) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Silu")
}
func (backend *Backend) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Swish")
}
func (backend *Backend) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "GeluTanh")
}
func (backend *Backend) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Gelu")
}
func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "ReLU")
}
func (backend *Backend) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "LeakyReLU")
}
func (backend *Backend) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "ELU")
}
func (backend *Backend) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "CELU")
}
func (backend *Backend) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "SELU")
}
func (backend *Backend) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Softplus")
}
func (backend *Backend) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Mish")
}
func (backend *Backend) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Softsign")
}
func (backend *Backend) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "HardSigmoid")
}
func (backend *Backend) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "HardSwish")
}
func (backend *Backend) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "HardTanh")
}
func (backend *Backend) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "HardGelu")
}
func (backend *Backend) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "QuickGelu")
}
func (backend *Backend) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "TanhShrink")
}
func (backend *Backend) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "Softmax")
}
func (backend *Backend) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "LogSoftmax")
}
func (backend *Backend) PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	backend.dispatchOp("Activation", "PReLU")
}
func (backend *Backend) PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int) {
	backend.dispatchOp("Activation", "PReLUV")
}
func (backend *Backend) LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	backend.dispatchOp("Activation", "LeakyReLUSlope")
}
func (backend *Backend) ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.dispatchOp("Activation", "ELUAlpha")
}
func (backend *Backend) CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.dispatchOp("Activation", "CELUAlpha")
}
func (backend *Backend) Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32) {
	backend.dispatchOp("Activation", "Threshold")
}
func (backend *Backend) HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32) {
	backend.dispatchOp("Activation", "HardTanhRange")
}
func (backend *Backend) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.dispatchOp("Activation", "Snake")
}
func (backend *Backend) SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32) {
	backend.dispatchOp("Activation", "SnakeParametric")
}
func (backend *Backend) HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	backend.dispatchOp("Activation", "HardShrink")
}
func (backend *Backend) SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	backend.dispatchOp("Activation", "SoftShrink")
}
func (backend *Backend) RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32) {
	backend.dispatchOp("Activation", "RReLU")
}
func (backend *Backend) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "GLU")
}
func (backend *Backend) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "GeGLU")
}
func (backend *Backend) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "GeGLUTanh")
}
func (backend *Backend) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "SwiGLU")
}
func (backend *Backend) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "ReGLU")
}
func (backend *Backend) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "SiGLU")
}
func (backend *Backend) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "GLUTensors")
}
func (backend *Backend) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "GeGLUTensors")
}
func (backend *Backend) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "GeGLUTanhTensors")
}
func (backend *Backend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "SwiGLUTensors")
}
func (backend *Backend) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "ReGLUTensors")
}
func (backend *Backend) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "SiGLUTensors")
}
func (backend *Backend) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "LinGLU")
}
func (backend *Backend) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.dispatchOp("Activation", "SeGLU")
}
func (backend *Backend) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "LinGLUTensors")
}
func (backend *Backend) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Activation", "SeGLUTensors")
}
func (backend *Backend) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Add")
}
func (backend *Backend) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Sub")
}
func (backend *Backend) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Mul")
}
func (backend *Backend) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Div")
}
func (backend *Backend) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Max")
}
func (backend *Backend) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Min")
}
func (backend *Backend) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Abs")
}
func (backend *Backend) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Neg")
}
func (backend *Backend) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Sqrt")
}
func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Elementwise", "ReLU")
}
func (backend *Backend) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	backend.dispatchOp("Elementwise", "Axpy")
}
func (backend *Backend) Sum(values unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Reduction", "Sum")
}
func (backend *Backend) Prod(values unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Reduction", "Prod")
}
func (backend *Backend) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Reduction", "ReduceMin")
}
func (backend *Backend) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Reduction", "ReduceMax")
}
func (backend *Backend) L1Norm(values unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Reduction", "L1Norm")
}
func (backend *Backend) Dot(left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Dot", "Dot")
}
func (backend *Backend) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Losses", "MSE")
}
func (backend *Backend) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Losses", "MAE")
}
func (backend *Backend) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Losses", "Huber")
}
func (backend *Backend) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Losses", "BinaryCrossEntropy")
}
func (backend *Backend) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Losses", "KLDivergence")
}
func (backend *Backend) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) {
	backend.dispatchOp("Sampling", "GreedySample")
}
func (backend *Backend) TopKSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) {
	backend.dispatchOp("Sampling", "TopKSample")
}
func (backend *Backend) TopPSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) {
	backend.dispatchOp("Sampling", "TopPSample")
}
func (backend *Backend) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "Laplacian")
}
func (backend *Backend) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "Laplacian4")
}
func (backend *Backend) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "Grad1D")
}
func (backend *Backend) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "Divergence1D")
}
func (backend *Backend) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Physics", "FFT1D")
}
func (backend *Backend) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Physics", "IFFT1D")
}
func (backend *Backend) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "QuantumPotential")
}
func (backend *Backend) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.dispatchOp("Physics", "BohmianVelocity")
}
func (backend *Backend) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	backend.dispatchOp("Causal", "Cholesky")
}
func (backend *Backend) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Causal", "CATE")
}
func (backend *Backend) ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("Masking", "ApplyMask")
}
func (backend *Backend) CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	backend.dispatchOp("Masking", "CausalMask")
}
func (backend *Backend) ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	backend.dispatchOp("Masking", "ALiBiBias")
}
func (backend *Backend) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("VSA", "Bind")
}
func (backend *Backend) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("VSA", "Bundle")
}
func (backend *Backend) Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("VSA", "Permute")
}
func (backend *Backend) InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("VSA", "InversePermute")
}
func (backend *Backend) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("VSA", "Similarity")
}
func (backend *Backend) BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("ActiveInference", "BeliefUpdate")
}
func (backend *Backend) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	backend.dispatchOp("ActiveInference", "PrecisionWeight")
}
func (backend *Backend) Dequant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	backend.dispatchOp("Dequant", "Dequant")
}
func (backend *Backend) Dequant4(dst, src unsafe.Pointer, pairCount int, config DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	backend.dispatchOp("Dequant", "Dequant4")
}
func (backend *Backend) Quant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	backend.dispatchOp("Quant", "Quant")
}
