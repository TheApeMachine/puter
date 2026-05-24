//go:build xla

package xla

import (
	"math"
	"math/rand/v2"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func (host *ComputeHost) DispatchTopKSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	if vocabSize == 0 || host.bridge == nil {
		return 0
	}

	sorted, indices := host.executeSoftmaxSort(logits, vocabSize, format, config.Temperature)
	topK := config.TopK

	if topK <= 0 || topK > vocabSize {
		topK = vocabSize
	}

	return selectTopKIndex(sorted, indices, topK, config.Seed)
}

func (host *ComputeHost) DispatchTopPSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	if vocabSize == 0 || host.bridge == nil {
		return 0
	}

	sorted, indices := host.executeSoftmaxSort(logits, vocabSize, format, config.Temperature)

	return selectTopPIndex(sorted, indices, config.TopP, config.Seed)
}

func (host *ComputeHost) executeSoftmaxSort(
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
	temperature float32,
) ([]float32, []int32) {
	inputShape, err := ShapeFromCount(vocabSize)
	host.dispatchError(err)

	stackShape, err := ShapeFromCount(vocabSize * 2)
	host.dispatchError(err)

	temperatureValue := temperature

	if temperatureValue == 0 {
		temperatureValue = 1
	}

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: format,
		OutputShape: stackShape,
	}

	stackTensor := host.borrowVectorBuffer(format, vocabSize*2)
	defer stackTensor.Close()

	inputTensor := host.requireDeviceTensor(logits)

	host.dispatchError(host.builder.ExecuteSoftmaxSort(
		host.bridge,
		context,
		[]float64{float64(temperatureValue)},
		inputTensor,
		stackTensor,
	))

	stacked := host.readVectorFloat32(stackTensor, vocabSize*2)
	sorted := stacked[:vocabSize]
	indices := make([]int32, vocabSize)

	for index := range indices {
		indices[index] = int32(math.Round(float64(stacked[vocabSize+index])))
	}

	return sorted, indices
}

func (host *ComputeHost) readVectorFloat32(deviceTensor *DeviceTensor, count int) []float32 {
	_, bytesOut, err := host.bridge.download(deviceTensor)
	host.dispatchError(err)

	decoded, err := convert.BytesToFloat32(deviceTensor.format(), bytesOut)
	host.dispatchError(err)

	if len(decoded) < count {
		host.dispatchError(&loweringError{message: "short XLA vector download"})
	}

	return decoded[:count]
}

func selectTopKIndex(sorted []float32, indices []int32, topK int, seed uint64) int32 {
	var sum float32

	for index := 0; index < topK; index++ {
		sum += sorted[index]
	}

	if sum == 0 {
		return indices[0]
	}

	rng := newSamplingRNG(seed)
	target := rng.Float32() * sum
	cumulative := float32(0)

	for index := 0; index < topK; index++ {
		cumulative += sorted[index]

		if cumulative >= target {
			return indices[index]
		}
	}

	return indices[topK-1]
}

func selectTopPIndex(sorted []float32, indices []int32, topP float32, seed uint64) int32 {
	if topP <= 0 {
		topP = 1
	}

	if topP > 1 {
		topP = 1
	}

	prefixLength := len(sorted)
	cumulative := float32(0)

	for index, probability := range sorted {
		cumulative += probability

		if cumulative >= topP {
			prefixLength = index + 1
			break
		}
	}

	if prefixLength == 0 {
		prefixLength = 1
	}

	var sum float32

	for index := 0; index < prefixLength; index++ {
		sum += sorted[index]
	}

	if sum == 0 {
		return indices[0]
	}

	rng := newSamplingRNG(seed)
	target := rng.Float32() * sum
	cumulative = 0

	for index := 0; index < prefixLength; index++ {
		cumulative += sorted[index]

		if cumulative >= target {
			return indices[index]
		}
	}

	return indices[prefixLength-1]
}

func newSamplingRNG(seed uint64) *rand.Rand {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source)
}
