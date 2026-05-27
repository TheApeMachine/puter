package embedding

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (embedding Embedding) TimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	dispatchTimestepEmbedding(config, timesteps, output, count, dim, format)
}

func dispatchTimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	if count == 0 || dim == 0 {
		return
	}

	if err := config.Validate(); err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		runTimestepEmbedding(config, timesteps, output, count, dim, format)
	default:
		panic("embedding.timestep: unsupported dtype")
	}
}

func runTimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	for rowIndex := range count {
		timestep := *(*float32)(unsafe.Add(timesteps, uintptr(rowIndex)*4))

		for dimIndex := range dim {
			value := timestepEmbeddingValue(config, timestep, dim, dimIndex)
			storeTimestepEmbeddingValue(output, rowIndex*dim+dimIndex, value, format)
		}
	}
}

func timestepEmbeddingValue(
	config device.TimestepEmbeddingConfig,
	timestep float32,
	dim int,
	dimIndex int,
) float32 {
	halfDim := dim / 2

	if halfDim == 0 || dimIndex >= halfDim*2 {
		return 0
	}

	firstHalf := dimIndex < halfDim
	frequencyIndex := dimIndex

	if !firstHalf {
		frequencyIndex -= halfDim
	}

	denominator := float32(halfDim) - config.DownscaleFreqShift
	exponent := -float32(math.Log(float64(config.MaxPeriod))) * float32(frequencyIndex) / denominator
	angle := float64((timestep / config.TimestepDivisor) * float32(math.Exp(float64(exponent))))
	sinValue := float32(math.Sin(angle))
	cosValue := float32(math.Cos(angle))

	if config.FlipSinToCos {
		if firstHalf {
			return cosValue
		}

		return sinValue
	}

	if firstHalf {
		return sinValue
	}

	return cosValue
}

func storeTimestepEmbeddingValue(
	output unsafe.Pointer,
	index int,
	value float32,
	format dtype.DType,
) {
	switch format {
	case dtype.Float32:
		*(*float32)(unsafe.Add(output, uintptr(index)*4)) = value
	case dtype.Float16:
		storeF16(output, index, value)
	case dtype.BFloat16:
		storeBF16(output, index, value)
	}
}
