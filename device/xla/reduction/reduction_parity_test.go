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

	// Zero-host-sync (ARCHITECTURE.md §2.2): the public Reduction methods
	// write into `dst`. We adapt them to the parity-test callback shape
	// (which still returns float32 for comparison) by allocating a stack
	// local. The reference path on CPU does the same internally via the
	// `*Native` helpers.
	runReduction := func(
		op func(dst, values unsafe.Pointer, count int, format dtype.DType),
		sourceTensor *xla.DeviceTensor, count int, format dtype.DType,
	) float32 {
		var result float32
		op(unsafe.Pointer(&result), unsafe.Pointer(sourceTensor), count, format)
		return result
	}

	expectReduction := func(
		op func(dst, values unsafe.Pointer, count int, format dtype.DType),
		values unsafe.Pointer, count int, format dtype.DType,
	) float32 {
		var result float32
		op(unsafe.Pointer(&result), values, count, format)
		return result
	}

	reductionCases := []struct {
		name   string
		run    func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32
		expect func(values unsafe.Pointer, count int, format dtype.DType) float32
	}{
		{
			name: "Sum",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runReduction(harness.Backend().Sum, sourceTensor, count, format)
			},
			expect: func(values unsafe.Pointer, count int, format dtype.DType) float32 {
				return expectReduction(referenceReduction.Sum, values, count, format)
			},
		},
		{
			name: "Prod",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runReduction(harness.Backend().Prod, sourceTensor, count, format)
			},
			expect: func(values unsafe.Pointer, count int, format dtype.DType) float32 {
				return expectReduction(referenceReduction.Prod, values, count, format)
			},
		},
		{
			name: "ReduceMin",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runReduction(harness.Backend().ReduceMin, sourceTensor, count, format)
			},
			expect: func(values unsafe.Pointer, count int, format dtype.DType) float32 {
				return expectReduction(referenceReduction.ReduceMin, values, count, format)
			},
		},
		{
			name: "ReduceMax",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runReduction(harness.Backend().ReduceMax, sourceTensor, count, format)
			},
			expect: func(values unsafe.Pointer, count int, format dtype.DType) float32 {
				return expectReduction(referenceReduction.ReduceMax, values, count, format)
			},
		},
		{
			name: "L1Norm",
			run: func(sourceTensor *xla.DeviceTensor, count int, format dtype.DType) float32 {
				return runReduction(harness.Backend().L1Norm, sourceTensor, count, format)
			},
			expect: func(values unsafe.Pointer, count int, format dtype.DType) float32 {
				return expectReduction(referenceReduction.L1Norm, values, count, format)
			},
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
