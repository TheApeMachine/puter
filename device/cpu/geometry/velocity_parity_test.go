package geometry

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/elementwise"
)

func TestPhaseVelocityFloat32Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseVelocityFloat32", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match elementwise.SubFloat32Native at N=%d", count), func() {
				current, previous, expected := phaseVelocityFloat32ReferenceBuffers(count)
				got := make([]float32, count)

				runPhaseVelocityFloat32(
					unsafePointerFromFloat32Slice(got),
					unsafePointerFromFloat32Slice(current),
					unsafePointerFromFloat32Slice(previous),
					count,
				)
				convey.So(got, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestPhaseVelocityFloat16Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseVelocityFloat16", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match elementwise.SubFloat16Native at N=%d", count), func() {
				current, previous, expected := phaseVelocityFloat16ReferenceBuffers(count)
				got := make([]uint16, count)

				runPhaseVelocityFloat16(
					unsafePointerFromUInt16Slice(got),
					unsafePointerFromUInt16Slice(current),
					unsafePointerFromUInt16Slice(previous),
					count,
				)
				convey.So(got, convey.ShouldResemble, expected)
			})
		}
	})
}

func TestPhaseVelocityBFloat16Parity(testingObject *testing.T) {
	convey.Convey("Given PhaseVelocityBFloat16", testingObject, func() {
		for _, count := range phaseCouplingParitySizes {
			convey.Convey(fmt.Sprintf("It should match elementwise.SubBFloat16Native at N=%d", count), func() {
				current, previous, expected := phaseVelocityBFloat16ReferenceBuffers(count)
				got := make([]uint16, count)

				runPhaseVelocityBFloat16(
					unsafePointerFromUInt16Slice(got),
					unsafePointerFromUInt16Slice(current),
					unsafePointerFromUInt16Slice(previous),
					count,
				)
				convey.So(got, convey.ShouldResemble, expected)
			})
		}
	})
}

func BenchmarkPhaseVelocityFloat32(benchmark *testing.B) {
	count := 8192
	current, previous, _ := phaseVelocityFloat32ReferenceBuffers(count)
	got := make([]float32, count)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		runPhaseVelocityFloat32(
			unsafePointerFromFloat32Slice(got),
			unsafePointerFromFloat32Slice(current),
			unsafePointerFromFloat32Slice(previous),
			count,
		)
	}
}

func phaseVelocityFloat32ReferenceBuffers(count int) ([]float32, []float32, []float32) {
	current := make([]float32, count)
	previous := make([]float32, count)
	expected := make([]float32, count)

	for index := 0; index < count; index++ {
		current[index] = float32(math.Sin(float64(index%17))) * 2.5
		previous[index] = float32(math.Cos(float64(index%13))) * 1.75
	}

	elementwise.SubFloat32Native(expected, current, previous)

	return current, previous, expected
}

func phaseVelocityFloat16ReferenceBuffers(count int) ([]uint16, []uint16, []uint16) {
	currentValues := make([]float32, count)
	previousValues := make([]float32, count)
	current := make([]dtype.F16, count)
	previous := make([]dtype.F16, count)
	expected := make([]dtype.F16, count)

	for index := 0; index < count; index++ {
		currentValues[index] = float32(math.Sin(float64(index%17))) * 2.5
		previousValues[index] = float32(math.Cos(float64(index%13))) * 1.75
		current[index] = dtype.Fromfloat32(currentValues[index])
		previous[index] = dtype.Fromfloat32(previousValues[index])
	}

	elementwise.SubFloat16Native(expected, current, previous)

	return f16SliceToUInt16(current), f16SliceToUInt16(previous), f16SliceToUInt16(expected)
}

func phaseVelocityBFloat16ReferenceBuffers(count int) ([]uint16, []uint16, []uint16) {
	currentValues := make([]float32, count)
	previousValues := make([]float32, count)
	current := make([]dtype.BF16, count)
	previous := make([]dtype.BF16, count)
	expected := make([]dtype.BF16, count)

	for index := 0; index < count; index++ {
		currentValues[index] = float32(math.Sin(float64(index%17))) * 2.5
		previousValues[index] = float32(math.Cos(float64(index%13))) * 1.75
		current[index] = dtype.NewBfloat16FromFloat32(currentValues[index])
		previous[index] = dtype.NewBfloat16FromFloat32(previousValues[index])
	}

	elementwise.SubBFloat16Native(expected, current, previous)

	return bf16SliceToUInt16(current), bf16SliceToUInt16(previous), bf16SliceToUInt16(expected)
}

func f16SliceToUInt16(values []dtype.F16) []uint16 {
	storage := make([]uint16, len(values))

	for index, value := range values {
		storage[index] = value.Bits()
	}

	return storage
}

func bf16SliceToUInt16(values []dtype.BF16) []uint16 {
	storage := make([]uint16, len(values))

	for index, value := range values {
		storage[index] = uint16(value)
	}

	return storage
}
