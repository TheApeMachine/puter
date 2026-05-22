package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
)

type swiGLUFixture struct {
	gateBytes     []byte
	upBytes       []byte
	expectedBytes []byte
}

func swiGLUFixtureForTest(elementCount int, storageDType dtype.DType) swiGLUFixture {
	gateValues := swiGLUGateValuesForTest(elementCount)
	upValues := swiGLUUpValuesForTest(elementCount)
	destination := swiGLUExpectedFloat32ForTest(gateValues, upValues)

	if storageDType == dtype.Float32 {
		return swiGLUFixture{
			gateBytes:     dtypeconvert.Float32ToBytes(gateValues),
			upBytes:       dtypeconvert.Float32ToBytes(upValues),
			expectedBytes: dtypeconvert.Float32ToBytes(destination),
		}
	}

	gateBytes := encodeFloat32ValuesAsDType(gateValues, storageDType)
	upBytes := encodeFloat32ValuesAsDType(upValues, storageDType)

	storedGate := decodeDTypeBytesToFloat32(gateBytes, storageDType)
	storedUp := decodeDTypeBytesToFloat32(upBytes, storageDType)
	roundTrip := swiGLUExpectedFloat32ForTest(storedGate, storedUp)

	return swiGLUFixture{
		gateBytes:     gateBytes,
		upBytes:       upBytes,
		expectedBytes: encodeFloat32ValuesAsDType(roundTrip, storageDType),
	}
}

func swiGLUGateValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index)*0.1 - 0.5
	}

	return values
}

func swiGLUUpValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index)*0.1 + 0.5
	}

	return values
}

func swiGLUExpectedFloat32ForTest(gateValues []float32, upValues []float32) []float32 {
	destination := make([]float32, len(gateValues))

	for index := range gateValues {
		silu := cpumath.FastSilu32(gateValues[index])
		destination[index] = normMetalFMAFloat32(silu, upValues[index], 0)
	}

	return destination
}
