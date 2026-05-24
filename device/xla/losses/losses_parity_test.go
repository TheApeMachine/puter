//go:build xla

package losses_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpulosses "github.com/theapemachine/puter/device/cpu/losses"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceLosses = cpulosses.New()

func TestLossesXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	pairCases := []struct {
		name   string
		run    func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32
		expect func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	}{
		{
			name: "MSE",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().MSE(
					xla.ResidentPointer(predictions),
					xla.ResidentPointer(targets),
					count,
					format,
				)
			},
			expect: referenceLosses.MSE,
		},
		{
			name: "MAE",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().MAE(
					xla.ResidentPointer(predictions),
					xla.ResidentPointer(targets),
					count,
					format,
				)
			},
			expect: referenceLosses.MAE,
		},
		{
			name: "Huber",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().Huber(
					xla.ResidentPointer(predictions),
					xla.ResidentPointer(targets),
					count,
					format,
				)
			},
			expect: referenceLosses.Huber,
		},
		{
			name: "BinaryCrossEntropy",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().BinaryCrossEntropy(
					xla.ResidentPointer(predictions),
					xla.ResidentPointer(targets),
					count,
					format,
				)
			},
			expect: referenceLosses.BinaryCrossEntropy,
		},
		{
			name: "KLDivergence",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().KLDivergence(
					xla.ResidentPointer(predictions),
					xla.ResidentPointer(targets),
					count,
					format,
				)
			},
			expect: referenceLosses.KLDivergence,
		},
	}

	for _, pairCase := range pairCases {
		pairCase := pairCase

		convey.Convey(fmt.Sprintf("Given XLA %s loss", pairCase.name), t, func() {
			for _, storageDType := range xlaparity.FloatParityDTypes {
				storageDType := storageDType

				convey.Convey(storageDType.Name(), func() {
					for _, count := range xlaparity.Lengths {
						convey.Convey(fmt.Sprintf("N=%d", count), func() {
							predictions := xlaparity.RandomUnaryInput(count, 0x3100+int64(count))
							targets := xlaparity.RandomUnaryInput(count, 0x3200+int64(count))

							want := pairCase.expect(
								unsafe.Pointer(&predictions[0]),
								unsafe.Pointer(&targets[0]),
								count,
								storageDType,
							)

							predictionsTensor := harness.UploadVector(predictions, storageDType)
							targetsTensor := harness.UploadVector(targets, storageDType)
							defer predictionsTensor.Close()
							defer targetsTensor.Close()

							got := pairCase.run(predictionsTensor, targetsTensor, count, storageDType)
							xlaparity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 4)
						})
					}
				})
			}
		})
	}

	convey.Convey("Given XLA cross entropy loss", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, batchSize := range []int{1, 7, 64} {
					classes := 16
					batchSize := batchSize

					convey.Convey(fmt.Sprintf("batch=%d classes=%d", batchSize, classes), func() {
						logits := xlaparity.RandomUnaryInput(batchSize*classes, 0x3300+int64(batchSize))
						targets := make([]int32, batchSize)

						for batchIndex := range targets {
							targets[batchIndex] = int32(batchIndex % classes)
						}

						targetBytes := make([]byte, len(targets)*4)

						for index, value := range targets {
							offset := index * 4
							targetBytes[offset] = byte(value)
							targetBytes[offset+1] = byte(value >> 8)
							targetBytes[offset+2] = byte(value >> 16)
							targetBytes[offset+3] = byte(value >> 24)
						}

						want := referenceLosses.CrossEntropy(
							unsafe.Pointer(&logits[0]),
							unsafe.Pointer(&targetBytes[0]),
							batchSize,
							classes,
							storageDType,
						)

						logitsTensor := harness.UploadMatrix(logits, batchSize, classes, storageDType)
						targetsTensor := harness.UploadInt32Vector(targets)
						defer logitsTensor.Close()
						defer targetsTensor.Close()

						got := harness.Backend().CrossEntropy(
							xla.ResidentPointer(logitsTensor),
							xla.ResidentPointer(targetsTensor),
							batchSize,
							classes,
							storageDType,
						)
						xlaparity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 4)
					})
				}
			})
		}
	})
}
