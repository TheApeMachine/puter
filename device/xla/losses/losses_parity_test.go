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

// Zero-host-sync (ARCHITECTURE.md §2.2): the public loss methods write
// into `dst`. The parity test compares the result as a float32 — we
// adapt by allocating a stack local and bridging through it.

func runPairLoss(
	op func(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType),
	predictions, targets unsafe.Pointer, count int, format dtype.DType,
) float32 {
	var result float32
	op(unsafe.Pointer(&result), predictions, targets, count, format)
	return result
}

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
				return runPairLoss(harness.Backend().MSE, xla.ResidentPointer(predictions), xla.ResidentPointer(targets), count, format)
			},
			expect: func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
				return runPairLoss(referenceLosses.MSE, predictions, targets, count, format)
			},
		},
		{
			name: "MAE",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runPairLoss(harness.Backend().MAE, xla.ResidentPointer(predictions), xla.ResidentPointer(targets), count, format)
			},
			expect: func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
				return runPairLoss(referenceLosses.MAE, predictions, targets, count, format)
			},
		},
		{
			name: "Huber",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runPairLoss(harness.Backend().Huber, xla.ResidentPointer(predictions), xla.ResidentPointer(targets), count, format)
			},
			expect: func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
				return runPairLoss(referenceLosses.Huber, predictions, targets, count, format)
			},
		},
		{
			name: "BinaryCrossEntropy",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runPairLoss(harness.Backend().BinaryCrossEntropy, xla.ResidentPointer(predictions), xla.ResidentPointer(targets), count, format)
			},
			expect: func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
				return runPairLoss(referenceLosses.BinaryCrossEntropy, predictions, targets, count, format)
			},
		},
		{
			name: "KLDivergence",
			run: func(predictions, targets *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runPairLoss(harness.Backend().KLDivergence, xla.ResidentPointer(predictions), xla.ResidentPointer(targets), count, format)
			},
			expect: func(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
				return runPairLoss(referenceLosses.KLDivergence, predictions, targets, count, format)
			},
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

						var want float32
						referenceLosses.CrossEntropy(
							unsafe.Pointer(&want),
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

						var got float32
						harness.Backend().CrossEntropy(
							unsafe.Pointer(&got),
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
