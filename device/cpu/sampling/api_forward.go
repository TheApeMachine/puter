package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultSampling = New()

func GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return defaultSampling.GreedySample(logits, vocabSize, format)
}

func TopKSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return defaultSampling.TopKSample(config, logits, vocabSize, format)
}

func TopPSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	return defaultSampling.TopPSample(config, logits, vocabSize, format)
}
