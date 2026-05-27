//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

func TestDeviceTensorRawBytesWaitsForCommands(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given a queued Metal matmul without an explicit device sync", testingObject, func() {
		rows := 32
		inner := 128
		cols := 32
		leftValues := patternedValues(rows*inner, 0.017)
		rightValues := patternedValues(inner*cols, -0.011)
		want := referenceMatmul(leftValues, rightValues, rows, inner, cols)
		left := uploadRoPETensor(testingObject, backend, leftValues)
		defer left.Close()
		right := uploadRoPETensor(testingObject, backend, rightValues)
		defer right.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, rows*cols))
		defer output.Close()

		backend.Matmul(
			output.DispatchPointer(),
			left.DispatchPointer(),
			right.DispatchPointer(),
			rows,
			inner,
			cols,
			dtype.Float32,
		)

		convey.Convey("It should return completed device results at the readback boundary", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 2)
		})
	})
}

func TestComputeHostMatmulLaunchRejectsShortOutput(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given a Metal matmul output tensor smaller than the launch shape", testingObject, func() {
		left := uploadRoPETensor(testingObject, backend, make([]float32, 6))
		defer left.Close()
		right := uploadRoPETensor(testingObject, backend, make([]float32, 12))
		defer right.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 4))
		defer output.Close()

		panicValue := capturePanic(func() {
			backend.Matmul(
				output.DispatchPointer(),
				left.DispatchPointer(),
				right.DispatchPointer(),
				2,
				3,
				4,
				dtype.Float32,
			)
		})

		convey.Convey("It should fail before encoding the Metal command", func() {
			convey.So(
				fmt.Sprint(panicValue),
				convey.ShouldContainSubstring,
				"output buffer has 16 bytes, need 32 bytes",
			)
		})
	})
}

func TestViewWorkspaceSlotKeepsResidentTensorLive(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given a Metal workspace slot view", testingObject, func() {
		slot, err := backend.AllocateWorkspaceSlot(64)
		convey.So(err, convey.ShouldBeNil)
		defer slot.Close()

		shape, err := tensor.NewShape([]int{4})
		convey.So(err, convey.ShouldBeNil)

		view, err := backend.ViewWorkspaceSlot(slot, shape, dtype.Float32, 16)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should remain dispatchable after runtime value cleanup", func() {
			convey.So(view.Close(), convey.ShouldBeNil)
			convey.So(view.(*DeviceTensor).DispatchPointer(), convey.ShouldNotBeNil)

			prefix, err := view.Slice(0, 2)
			convey.So(err, convey.ShouldBeNil)
			convey.So(prefix.Bytes(), convey.ShouldEqual, 8)
			convey.So(prefix.(*DeviceTensor).DispatchPointer(), convey.ShouldNotBeNil)
		})
	})
}

func BenchmarkDeviceTensorRawBytesAfterMatmul(benchmark *testing.B) {
	backend := newMetalBenchmarkBackend(benchmark)
	defer backend.Close()

	rows := 16
	inner := 128
	cols := 16
	left := uploadRoPETensor(benchmark, backend, patternedValues(rows*inner, 0.017))
	defer left.Close()
	right := uploadRoPETensor(benchmark, backend, patternedValues(inner*cols, -0.011))
	defer right.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, rows*cols))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.Matmul(
			output.DispatchPointer(),
			left.DispatchPointer(),
			right.DispatchPointer(),
			rows,
			inner,
			cols,
			dtype.Float32,
		)

		if _, _, err := output.RawBytes(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func capturePanic(call func()) any {
	var panicValue any

	func() {
		defer func() {
			panicValue = recover()
		}()

		call()
	}()

	return panicValue
}

func patternedValues(count int, scale float32) []float32 {
	values := make([]float32, count)

	for index := range values {
		centered := float32(index%23) - 11.0
		values[index] = centered * scale
	}

	return values
}
