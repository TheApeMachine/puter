//go:build xla

package reduction_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpureduction "github.com/theapemachine/puter/device/cpu/reduction"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceReduction = cpureduction.New()

func TestReductionXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	reductionCases := []struct {
		name   string
		run    func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32
		expect func(values unsafe.Pointer, count int, format dtype.DType) float32
	}{
		{
			name: "Sum",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().Sum(unsafe.Pointer(sourceTensor), count, format)
			},
			expect: referenceReduction.Sum,
		},
		{
			name: "Prod",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().Prod(unsafe.Pointer(sourceTensor), count, format)
			},
			expect: referenceReduction.Prod,
		},
		{
			name: "ReduceMin",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().ReduceMin(unsafe.Pointer(sourceTensor), count, format)
			},
			expect: referenceReduction.ReduceMin,
		},
		{
			name: "ReduceMax",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().ReduceMax(unsafe.Pointer(sourceTensor), count, format)
			},
			expect: referenceReduction.ReduceMax,
		},
		{
			name: "L1Norm",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return harness.Backend().L1Norm(unsafe.Pointer(sourceTensor), count, format)
			},
			expect: referenceReduction.L1Norm,
		},
	}

	for _, reductionCase := range reductionCases {
		reductionCase := reductionCase

		convey.Convey(fmt.Sprintf("Given XLA %s reduction", reductionCase.name), t, func() {
			for _, storageDType := range xlaparity.FloatParityDTypes {
				storageDType := storageDType

				convey.Convey(storageDType.Name(), func() {
					for _, count := range xlaparity.Lengths {
						convey.Convey(fmt.Sprintf("N=%d", count), func() {
							source := xlaparity.RandomUnaryInput(count, 0x1200+int64(count))
							sourceBytes, err := xlaparity.EncodeVector(source, storageDType)

							if err != nil {
								t.Fatalf("encode source: %v", err)
							}

							want := reductionCase.expect(
								unsafe.Pointer(&sourceBytes[0]),
								count,
								storageDType,
							)

							sourceTensor := harness.UploadVector(source, storageDType)
							defer sourceTensor.Close()

							got := reductionCase.run(sourceTensor, count, storageDType)
							xlaparity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 2)
						})
					}
				})
			}
		})
	}
}
