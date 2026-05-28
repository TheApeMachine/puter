//go:build xla

package xla

import (
	"math/rand/v2"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func (host *ComputeHost) DispatchTopKSample(
	dst unsafe.Pointer,
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	if vocabSize == 0 || host.bridge == nil {
		return
	}

	topK := config.TopK

	if topK <= 0 || topK > vocabSize {
		topK = vocabSize
	}

	host.dispatchSample(dst, logits, vocabSize, format, "topk_sample", config.Temperature, topK, 0, config.Seed)
}

func (host *ComputeHost) DispatchTopPSample(
	dst unsafe.Pointer,
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	if vocabSize == 0 || host.bridge == nil {
		return
	}

	topP := config.TopP

	if topP <= 0 {
		topP = 1
	}

	if topP > 1 {
		topP = 1
	}

	host.dispatchSample(dst, logits, vocabSize, format, "topp_sample", config.Temperature, 0, topP, config.Seed)
}

func (host *ComputeHost) dispatchSample(
	dst unsafe.Pointer,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
	operationName string,
	temperature float32,
	topK int,
	topP float32,
	seed uint64,
) {
	inputShape, err := ShapeFromCount(vocabSize)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	temperatureValue := temperature

	if temperatureValue == 0 {
		temperatureValue = 1
	}

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: dtype.Int32,
		OutputShape: scalarShape,
	}

	inputTensor := host.requireDeviceTensor(logits)
	outputTensor := host.requireDeviceTensor(dst)
	randomTarget := newSamplingRNG(seed).Float32()

	host.dispatchError(host.builder.ExecuteProbabilisticSample(
		host.bridge,
		operationName,
		context,
		[]float64{float64(temperatureValue), float64(randomTarget), float64(topP)},
		[]int64{int64(topK)},
		inputTensor,
		outputTensor,
	))
}

func newSamplingRNG(seed uint64) *rand.Rand {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source)
}
