//go:build darwin && cgo

package model_editing

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpumodel "github.com/theapemachine/puter/device/cpu/model_editing"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestWeightGraftAddMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal WeightGraftAdd kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						weights := parity.RandomUnaryInput(count, 0x5300+int64(count))
						injection := parity.RandomUnaryInput(count, 0x5301+int64(count))
						weights = roundTripStorageValues(harness, weights, storageDType)
						injection = roundTripStorageValues(harness, injection, storageDType)
						want := graftReference(weights, injection, storageDType)

						weightsTensor := harness.UploadVector(weights, storageDType)
						injectionTensor := harness.UploadVector(injection, storageDType)
						defer weightsTensor.Close()
						defer injectionTensor.Close()

						dispatchErr := DispatchWeightGraftAddRefs(
							harness.ContextRef(),
							weightsTensor.Ref(),
							injectionTensor.Ref(),
							storageDType,
							uint32(count),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(weightsTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, graftMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func BenchmarkWeightGraftAddMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	count := 8192
	weights := parity.RandomUnaryInput(count, 0x5310)
	injection := parity.RandomUnaryInput(count, 0x5311)
	weightsTensor := harness.UploadVector(weights, dtype.Float32)
	injectionTensor := harness.UploadVector(injection, dtype.Float32)
	defer weightsTensor.Close()
	defer injectionTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchWeightGraftAddRefs(
			harness.ContextRef(),
			weightsTensor.Ref(),
			injectionTensor.Ref(),
			dtype.Float32,
			uint32(count),
		)
	}

	harness.Sync()
}

func graftReference(weights, injection []float32, format dtype.DType) []float32 {
	switch format {
	case dtype.Float32:
		got := append([]float32(nil), weights...)
		cpumodel.WeightGraftAddFloat32Native(got, injection)

		return got
	case dtype.Float16:
		got := append([]float32(nil), weights...)

		for index := range got {
			got[index] = dtype.Fromfloat32(weights[index] + injection[index]).Float32()
		}

		return got
	case dtype.BFloat16:
		got := make([]float32, len(weights))

		for index := range weights {
			got[index] = dtype.NewBfloat16FromFloat32(weights[index] + injection[index]).Float32()
		}

		return got
	default:
		panic(fmt.Sprintf("unsupported dtype %v", format))
	}
}

func graftMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return 0
	case dtype.Float16, dtype.BFloat16:
		return 1
	default:
		return 0
	}
}

func roundTripStorageValues(harness *parity.Harness, values []float32, format dtype.DType) []float32 {
	if format == dtype.Float32 {
		return values
	}

	buffer := harness.UploadVector(values, format)
	defer buffer.Close()

	return harness.DownloadFloat32(buffer, format)
}
