package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/dropout"
	"github.com/theapemachine/puter/device/cpu/losses"
	"github.com/theapemachine/puter/device/cpu/sampling"
)

func (backend *Backend) Dropout(
	dst, src unsafe.Pointer,
	count int,
	config device.DropoutConfig,
	format dtype.DType,
) {
	dropout.Dropout(dst, src, count, dropoutConfig(config), format)
}

func (backend *Backend) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return losses.MSE(predictions, targets, count, format)
}

func (backend *Backend) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return losses.MAE(predictions, targets, count, format)
}

func (backend *Backend) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return losses.Huber(predictions, targets, count, format)
}

func (backend *Backend) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return losses.BinaryCrossEntropy(predictions, targets, count, format)
}

func (backend *Backend) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return losses.KLDivergence(predictions, targets, count, format)
}

func (backend *Backend) CrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	return losses.CrossEntropy(logits, targets, batchSize, classes, format)
}

func (backend *Backend) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return sampling.GreedySample(logits, vocabSize, format)
}

func (backend *Backend) TopKSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return sampling.TopKSample(samplingConfig(config), logits, vocabSize, format)
}

func (backend *Backend) TopPSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return sampling.TopPSample(samplingConfig(config), logits, vocabSize, format)
}
