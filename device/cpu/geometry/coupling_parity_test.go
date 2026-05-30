package geometry

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

var phaseCouplingParitySizes = []int{1, 7, 64, 1024, 8192}

func TestPhaseCouplingFloat32Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseCouplingFloat32 kernels", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftValues, rightValues, expected := phaseCouplingReferenceBuffers(count)
				got := make([]float32, count)

				phaseCouplingFloat32Kernel(got, leftValues, rightValues, count)
				convey.So(got, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestPhaseCouplingFloat32NEONParity(testingObject *testing.T) {
	if !neonPhaseCouplingAvailable() {
		testingObject.Skip("NEON phase coupling unavailable")
	}

	convey.Convey("Given PhaseCouplingFloat32NEON", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftValues, rightValues, expected := phaseCouplingReferenceBuffers(count)
				got := make([]float32, count)

				PhaseCouplingFloat32NEON(got, leftValues, rightValues, count)
				cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, expected, 2)
			})
		}
	})
}

func TestPhaseCouplingFloat16Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseCouplingFloat16 kernels", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftStorage, rightStorage, expectedStorage := phaseCouplingFloat16ReferenceBuffers(count)
				gotStorage := make([]uint16, count)

				phaseCouplingFloat16Kernel(gotStorage, leftStorage, rightStorage, count)

				for index := 0; index < count; index++ {
					gotValue := dtype.F16(gotStorage[index]).Float32()
					expectedValue := dtype.F16(expectedStorage[index]).Float32()
					convey.So(
						float64(gotValue),
						convey.ShouldAlmostEqual,
						float64(expectedValue),
						1e-3,
					)
				}
			})
		}
	})
}

func TestPhaseCouplingBFloat16Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseCouplingBFloat16 kernels", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftStorage, rightStorage, expectedStorage := phaseCouplingBFloat16ReferenceBuffers(count)
				gotStorage := make([]uint16, count)

				phaseCouplingBFloat16Kernel(gotStorage, leftStorage, rightStorage, count)

				for index := 0; index < count; index++ {
					gotValue := dtype.BF16(gotStorage[index]).Float32()
					expectedValue := dtype.BF16(expectedStorage[index]).Float32()
					convey.So(
						float64(gotValue),
						convey.ShouldAlmostEqual,
						float64(expectedValue),
						1e-2,
					)
				}
			})
		}
	})
}

func TestPhaseCouplingFloat16NEONParity(testingObject *testing.T) {
	if !neonPhaseCouplingAvailable() {
		testingObject.Skip("NEON phase coupling unavailable")
	}

	convey.Convey("Given PhaseCouplingFloat16NEON", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftStorage, rightStorage, expectedStorage := phaseCouplingFloat16ReferenceBuffers(count)
				gotStorage := make([]uint16, count)

				PhaseCouplingFloat16NEON(gotStorage, leftStorage, rightStorage, count)

				for index := 0; index < count; index++ {
					gotValue := dtype.F16(gotStorage[index]).Float32()
					expectedValue := dtype.F16(expectedStorage[index]).Float32()
					convey.So(
						float64(gotValue),
						convey.ShouldAlmostEqual,
						float64(expectedValue),
						1e-3,
					)
				}
			})
		}
	})
}

func TestPhaseCouplingBFloat16NEONParity(testingObject *testing.T) {
	if !neonPhaseCouplingAvailable() {
		testingObject.Skip("NEON phase coupling unavailable")
	}

	convey.Convey("Given PhaseCouplingBFloat16NEON", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				leftStorage, rightStorage, expectedStorage := phaseCouplingBFloat16ReferenceBuffers(count)
				gotStorage := make([]uint16, count)

				PhaseCouplingBFloat16NEON(gotStorage, leftStorage, rightStorage, count)

				for index := 0; index < count; index++ {
					gotValue := dtype.BF16(gotStorage[index]).Float32()
					expectedValue := dtype.BF16(expectedStorage[index]).Float32()
					convey.So(
						float64(gotValue),
						convey.ShouldAlmostEqual,
						float64(expectedValue),
						1e-2,
					)
				}
			})
		}
	})
}

func BenchmarkPhaseCouplingFloat32Kernel(benchmark *testing.B) {
	count := 8192
	leftValues, rightValues, _ := phaseCouplingReferenceBuffers(count)
	got := make([]float32, count)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		phaseCouplingFloat32Kernel(got, leftValues, rightValues, count)
	}
}

func phaseCouplingFloat16ReferenceBuffers(count int) ([]uint16, []uint16, []uint16) {
	leftValues := make([]float32, count)
	rightValues := make([]float32, count)
	leftStorage := make([]uint16, count)
	rightStorage := make([]uint16, count)
	expected := make([]uint16, count)

	for index := 0; index < count; index++ {
		leftValues[index] = float32(math.Sin(float64(index%17))) * 2.5
		rightValues[index] = float32(math.Cos(float64(index%13))) * 1.75
		leftF16 := dtype.Fromfloat32(leftValues[index])
		rightF16 := dtype.Fromfloat32(rightValues[index])
		leftStorage[index] = uint16(leftF16)
		rightStorage[index] = uint16(rightF16)
		expected[index] = uint16(
			scalarPhaseCouplingReferenceF16(leftF16, rightF16),
		)
	}

	return leftStorage, rightStorage, expected
}

func phaseCouplingBFloat16ReferenceBuffers(count int) ([]uint16, []uint16, []uint16) {
	leftValues := make([]float32, count)
	rightValues := make([]float32, count)
	leftStorage := make([]uint16, count)
	rightStorage := make([]uint16, count)
	expected := make([]uint16, count)

	for index := 0; index < count; index++ {
		leftValues[index] = float32(math.Sin(float64(index%17))) * 2.5
		rightValues[index] = float32(math.Cos(float64(index%13))) * 1.75
		leftBF16 := dtype.NewBfloat16FromFloat32(leftValues[index])
		rightBF16 := dtype.NewBfloat16FromFloat32(rightValues[index])
		leftStorage[index] = uint16(leftBF16)
		rightStorage[index] = uint16(rightBF16)
		expected[index] = uint16(
			scalarPhaseCouplingReferenceBF16(leftBF16, rightBF16),
		)
	}

	return leftStorage, rightStorage, expected
}

func phaseCouplingReferenceBuffers(count int) ([]float32, []float32, []float32) {
	leftValues := make([]float32, count)
	rightValues := make([]float32, count)
	expected := make([]float32, count)

	for index := 0; index < count; index++ {
		leftValues[index] = float32(math.Sin(float64(index%17))) * 2.5
		rightValues[index] = float32(math.Cos(float64(index%13))) * 1.75
		expected[index] = scalarPhaseCouplingReference(leftValues[index], rightValues[index])
	}

	return leftValues, rightValues, expected
}

func float32SliceToF16(values []float32) []uint16 {
	storage := make([]uint16, len(values))

	for index, value := range values {
		storage[index] = uint16(dtype.Fromfloat32(value))
	}

	return storage
}

func float32SliceToBF16(values []float32) []uint16 {
	storage := make([]uint16, len(values))

	for index, value := range values {
		storage[index] = uint16(dtype.NewBfloat16FromFloat32(value))
	}

	return storage
}
