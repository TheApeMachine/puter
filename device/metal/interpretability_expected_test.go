package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	cpuinterpretability "github.com/theapemachine/puter/device/cpu/interpretability"
)

type activationSteerFixture struct {
	baseBytes        []byte
	directionBytes   []byte
	coefficientBytes []byte
	expectedBytes    []byte
}

func activationSteerFixtureForTest(elementCount int, storageDType dtype.DType) activationSteerFixture {
	base := activationSteerValuesForTest(elementCount, 3)
	direction := activationSteerValuesForTest(elementCount, 11)
	coefficient := float32(0.375)

	baseBytes := encodeLossValuesAsDType(base, storageDType)
	directionBytes := encodeLossValuesAsDType(direction, storageDType)

	storedBase := decodeDTypeBytesToFloat32(baseBytes, storageDType)
	storedDirection := decodeDTypeBytesToFloat32(directionBytes, storageDType)

	destination := make([]float32, elementCount)
	cpuinterpretability.ActivationSteerFloat32Scalar(
		destination, storedBase, storedDirection, coefficient,
	)

	return activationSteerFixture{
		baseBytes:        baseBytes,
		directionBytes:   directionBytes,
		coefficientBytes: encodeLossValuesAsDType([]float32{coefficient}, dtype.Float32),
		expectedBytes:    encodeLossValuesAsDType(destination, storageDType),
	}
}

func activationSteerValuesForTest(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*13+salt, 97, 23)
	}

	return values
}
