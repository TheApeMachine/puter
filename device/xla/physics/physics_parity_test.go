//go:build xla

package physics_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuphysics "github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referencePhysics = cpuphysics.New()

func TestPhysicsXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	spacing := float32(0.25)

	convey.Convey("Given XLA Grad1D", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0xc100+int64(count))
				want := make([]float32, count)
				referencePhysics.Grad1D(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Grad1D(
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-1", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				runLaplacianParity(t, harness, []int{count}, spacing, 0xd100+int64(count))
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-2", t, func() {
		for _, side := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("%dx%d", side, side), func() {
				runLaplacianParity(t, harness, []int{side, side}, spacing, 0xd200+int64(side))
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-3", t, func() {
		for _, depth := range []int{1, 4, 7} {
			for _, side := range []int{1, 7, 64} {
				convey.Convey(fmt.Sprintf("%dx%dx%d", depth, side, side), func() {
					runLaplacianParity(
						t,
						harness,
						[]int{depth, side, side},
						spacing,
						0xd300+int64(depth*100+side),
					)
				})
			}
		}
	})

	convey.Convey("Given XLA Laplacian4", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0xd400+int64(count))
				want := make([]float32, count)
				referencePhysics.Laplacian4(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Laplacian4(
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})
}

func runLaplacianParity(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	dims []int,
	spacing float32,
	seed int64,
) {
	testingTB.Helper()

	elementCount := 1

	for _, dim := range dims {
		elementCount *= dim
	}

	input := xlaparity.RandomUnaryInput(elementCount, seed)
	want := make([]float32, elementCount)
	referencePhysics.Laplacian(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&want[0]),
		dims,
		spacing,
		dtype.Float32,
	)

	var inputTensor *xla.DeviceTensor
	var outputTensor *xla.DeviceTensor

	switch len(dims) {
	case 1:
		inputTensor = harness.UploadVector(input, dtype.Float32)
		outputTensor = harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	case 2:
		inputTensor = harness.UploadMatrix(input, dims[0], dims[1], dtype.Float32)
		outputTensor = harness.UploadMatrix(make([]float32, elementCount), dims[0], dims[1], dtype.Float32)
	case 3:
		inputTensor = harness.UploadVolume(input, dims[0], dims[1], dims[2], dtype.Float32)
		outputTensor = harness.UploadVolume(make([]float32, elementCount), dims[0], dims[1], dims[2], dtype.Float32)
	default:
		testingTB.Fatalf("unsupported laplacian rank %d", len(dims))
	}

	defer inputTensor.Close()
	defer outputTensor.Close()

	harness.Backend().Laplacian(
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(outputTensor),
		dims,
		spacing,
		dtype.Float32,
	)

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func BenchmarkGrad1DXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	spacing := float32(0.25)
	input := xlaparity.RandomUnaryInput(count, 0xc200)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Grad1D(
			xla.ResidentPointer(inputTensor),
			xla.ResidentPointer(outputTensor),
			count,
			spacing,
			dtype.Float32,
		)
	}
}

func BenchmarkLaplacian2DXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	side := 64
	spacing := float32(0.25)
	elementCount := side * side
	input := xlaparity.RandomUnaryInput(elementCount, 0xd500)
	inputTensor := harness.UploadMatrix(input, side, side, dtype.Float32)
	outputTensor := harness.UploadMatrix(make([]float32, elementCount), side, side, dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Laplacian(
			xla.ResidentPointer(inputTensor),
			xla.ResidentPointer(outputTensor),
			[]int{side, side},
			spacing,
			dtype.Float32,
		)
	}
}
