//go:build xla

package vsa_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuvsa "github.com/theapemachine/puter/device/cpu/vsa"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceVSA = cpuvsa.New()

func TestVSAXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA VSA Bind", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := xlaparity.RandomUnaryInput(count, 0x6100+int64(count))
				right := xlaparity.RandomUnaryInput(count, 0x6200+int64(count))
				want := make([]float32, count)
				referenceVSA.Bind(
					unsafe.Pointer(&left[0]),
					unsafe.Pointer(&right[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Bind(
					xla.ResidentPointer(leftTensor),
					xla.ResidentPointer(rightTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA VSA Permute", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0x6300+int64(count))
				want := make([]float32, count)
				referenceVSA.Permute(
					device.VSAConfig{Shift: 3},
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Permute(
					device.VSAConfig{Shift: 3},
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA VSA InversePermute", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0x6350+int64(count))
				want := make([]float32, count)
				referenceVSA.InversePermute(
					device.VSAConfig{Shift: 3},
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().InversePermute(
					device.VSAConfig{Shift: 3},
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA VSA Bundle", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := xlaparity.RandomUnaryInput(count, 0x6600+int64(count))
				right := xlaparity.RandomUnaryInput(count, 0x6700+int64(count))
				want := make([]float32, count)
				referenceVSA.Bundle(
					unsafe.Pointer(&left[0]),
					unsafe.Pointer(&right[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Bundle(
					xla.ResidentPointer(leftTensor),
					xla.ResidentPointer(rightTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA VSA Similarity", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := xlaparity.RandomUnaryInput(count, 0x6800+int64(count))
				right := xlaparity.RandomUnaryInput(count, 0x6900+int64(count))
				var want float32
				referenceVSA.Similarity(
					unsafe.Pointer(&want),
					unsafe.Pointer(&left[0]),
					unsafe.Pointer(&right[0]),
					count,
					dtype.Float32,
				)

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()

				var got float32
				harness.Backend().Similarity(
					unsafe.Pointer(&got),
					xla.ResidentPointer(leftTensor),
					xla.ResidentPointer(rightTensor),
					count,
					dtype.Float32,
				)

				xlaparity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 4)
			})
		}
	})
}

func BenchmarkVSABindXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	left := xlaparity.RandomUnaryInput(count, 0x6400)
	right := xlaparity.RandomUnaryInput(count, 0x6500)
	leftTensor := harness.UploadVector(left, dtype.Float32)
	rightTensor := harness.UploadVector(right, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer leftTensor.Close()
	defer rightTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Bind(
			xla.ResidentPointer(leftTensor),
			xla.ResidentPointer(rightTensor),
			xla.ResidentPointer(outputTensor),
			count,
			dtype.Float32,
		)
	}
}
