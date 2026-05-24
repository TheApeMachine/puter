//go:build xla

package masking_test

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	cpumasking "github.com/theapemachine/puter/device/cpu/masking"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceMasking = cpumasking.New()

func TestMaskingXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA ApplyMask", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						input := xlaparity.RandomUnaryInput(count, 0x5100+int64(count))
						mask := xlaparity.RandomUnaryInput(count, 0x5200+int64(count))
						want := make([]float32, count)
						referenceMasking.ApplyMask(
							unsafe.Pointer(&input[0]),
							unsafe.Pointer(&mask[0]),
							unsafe.Pointer(&want[0]),
							count,
							storageDType,
						)

						inputTensor := harness.UploadVector(input, storageDType)
						maskTensor := harness.UploadVector(mask, storageDType)
						outputTensor := harness.UploadVector(make([]float32, count), storageDType)
						defer inputTensor.Close()
						defer maskTensor.Close()
						defer outputTensor.Close()

						harness.Backend().ApplyMask(
							xla.ResidentPointer(inputTensor),
							xla.ResidentPointer(maskTensor),
							xla.ResidentPointer(outputTensor),
							count,
							storageDType,
						)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
					})
				}
			})
		}
	})

	convey.Convey("Given XLA CausalMask", t, func() {
		sides := []int{1, 7, 64}

		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, side := range sides {
					convey.Convey(fmt.Sprintf("side=%d", side), func() {
						want := make([]float32, side*side)
						referenceMasking.CausalMask(
							unsafe.Pointer(&want[0]),
							side, side,
							storageDType,
						)

						outputTensor := harness.UploadMatrix(make([]float32, side*side), side, side, storageDType)
						defer outputTensor.Close()

						harness.Backend().CausalMask(
							xla.ResidentPointer(outputTensor),
							side, side,
							storageDType,
						)

						got := harness.DownloadFloat32(outputTensor, storageDType)

						for index := range want {
							if math.IsInf(float64(want[index]), -1) {
								convey.So(math.IsInf(float64(got[index]), -1), convey.ShouldBeTrue)
								continue
							}

							xlaparity.AssertFloat32SlicesWithinULP(
								t, []float32{got[index]}, []float32{want[index]}, 2,
							)
						}
					})
				}
			})
		}
	})

	convey.Convey("Given XLA ALiBiBias", t, func() {
		sides := []int{1, 7, 64}

		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, side := range sides {
					convey.Convey(fmt.Sprintf("side=%d", side), func() {
						scores := xlaparity.RandomUnaryInput(side*side, 0x5300+int64(side))
						slope := xlaparity.RandomUnaryInput(1, 0x5400+int64(side))
						want := make([]float32, side*side)
						referenceMasking.ALiBiBias(
							unsafe.Pointer(&scores[0]),
							unsafe.Pointer(&slope[0]),
							unsafe.Pointer(&want[0]),
							side, side,
							storageDType,
						)

						scoresTensor := harness.UploadMatrix(scores, side, side, storageDType)
						slopeTensor := harness.UploadVector(slope, storageDType)
						outputTensor := harness.UploadMatrix(make([]float32, side*side), side, side, storageDType)
						defer scoresTensor.Close()
						defer slopeTensor.Close()
						defer outputTensor.Close()

						harness.Backend().ALiBiBias(
							xla.ResidentPointer(scoresTensor),
							xla.ResidentPointer(slopeTensor),
							xla.ResidentPointer(outputTensor),
							side, side,
							storageDType,
						)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
					})
				}
			})
		}
	})
}
