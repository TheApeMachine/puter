package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	cpumodelediting "github.com/theapemachine/puter/device/cpu/model_editing"
)

type weightGraftFixture struct {
	weightsBytes   []byte
	injectionBytes []byte
	expectedBytes  []byte
}

func weightGraftFixtureForTest(elementCount int, storageDType dtype.DType) weightGraftFixture {
	weights := weightGraftValuesForTest(elementCount, 5)
	injection := weightGraftValuesForTest(elementCount, 17)

	weightsBytes := encodeLossValuesAsDType(weights, storageDType)
	injectionBytes := encodeLossValuesAsDType(injection, storageDType)

	storedWeights := decodeDTypeBytesToFloat32(weightsBytes, storageDType)
	storedInjection := decodeDTypeBytesToFloat32(injectionBytes, storageDType)

	want := make([]float32, elementCount)
	copy(want, storedWeights)
	cpumodelediting.WeightGraftAddFloat32Scalar(want, storedInjection)

	return weightGraftFixture{
		weightsBytes:   weightsBytes,
		injectionBytes: injectionBytes,
		expectedBytes:  encodeLossValuesAsDType(want, storageDType),
	}
}

func weightGraftValuesForTest(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*19+salt, 101, 29)
	}

	return values
}
