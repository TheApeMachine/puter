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

	convey.Convey("Given XLA Grad1D", t, func() {
		spacing := float32(0.25)

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
