//go:build darwin && cgo

package math

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestInvSqrtDimScaleMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal InvSqrtDimScale kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						input := parity.RandomUnaryInput(count, 0x5200+int64(count))
						input = roundTripStorageValues(harness, input, storageDType)
						want := make([]float32, count)
						dim := int32(64)

						invSqrtDimScaleReference(want, input, dim, storageDType)

						inputTensor := harness.UploadVector(input, storageDType)
						outputTensor := harness.UploadVector(make([]float32, count), storageDType)
						dimTensor := harness.UploadInt32([]int32{dim})
						defer inputTensor.Close()
						defer outputTensor.Close()
						defer dimTensor.Close()

						dispatchErr := DispatchInvSqrtDimScaleRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							dimTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(count),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, mathMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func TestLogSumExpMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal LogSumExp kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						cols := logSumExpCols(count)
						rows := count / cols
						input := parity.RandomUnaryInput(rows*cols, 0x5210+int64(count))
						input = roundTripStorageValues(harness, input, storageDType)
						want := make([]float32, rows)

						logSumExpReference(input, cols, want, storageDType)

						inputTensor := harness.UploadVector(input, storageDType)
						outputTensor := harness.UploadVector(make([]float32, rows), storageDType)
						defer inputTensor.Close()
						defer outputTensor.Close()

						dispatchErr := DispatchLogSumExpRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(rows),
							uint32(cols),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, mathMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func TestOuterMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal Outer kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						leftCount, rightCount := outerDims(count)
						left := parity.RandomUnaryInput(leftCount, 0x5220+int64(count))
						right := parity.RandomUnaryInput(rightCount, 0x5221+int64(count))
						left = roundTripStorageValues(harness, left, storageDType)
						right = roundTripStorageValues(harness, right, storageDType)
						want := make([]float32, leftCount*rightCount)

						outerReference(left, right, want, storageDType)

						leftTensor := harness.UploadVector(left, storageDType)
						rightTensor := harness.UploadVector(right, storageDType)
						outputTensor := harness.UploadVector(make([]float32, len(want)), storageDType)
						defer leftTensor.Close()
						defer rightTensor.Close()
						defer outputTensor.Close()

						dispatchErr := DispatchOuterRefs(
							harness.ContextRef(),
							leftTensor.Ref(),
							rightTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(leftCount),
							uint32(rightCount),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, mathMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func BenchmarkInvSqrtDimScaleMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	count := 8192
	input := parity.RandomUnaryInput(count, 0x5230)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	dimTensor := harness.UploadInt32([]int32{64})
	defer inputTensor.Close()
	defer outputTensor.Close()
	defer dimTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchInvSqrtDimScaleRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			dimTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(count),
		)
	}

	harness.Sync()
}

func logSumExpCols(count int) int {
	switch {
	case count <= 1:
		return 1
	case count <= 7:
		return count
	case count <= 64:
		return 8
	case count <= 1024:
		return 32
	default:
		return 128
	}
}

func outerDims(count int) (int, int) {
	leftCount := 1

	for leftCount*leftCount < count {
		leftCount++
	}

	rightCount := count / leftCount

	if rightCount == 0 {
		rightCount = 1
	}

	return leftCount, rightCount
}

func mathMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return 1
	case dtype.Float16, dtype.BFloat16:
		return 1
	default:
		return 1
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

func invSqrtDimScaleReference(out, input []float32, dim int32, format dtype.DType) {
	if format == dtype.Float32 {
		cpumath.InvSqrtDimScaleGeneric(out, input, dim)
		return
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	for index, value := range input {
		out[index] = storeMathResult(value*scale, format)
	}
}

func logSumExpReference(input []float32, cols int, out []float32, format dtype.DType) {
	if format == dtype.Float32 {
		cpumath.LogSumExpGeneric(input, cols, out)
		return
	}

	rows := len(input) / cols

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * cols
		row := input[rowOffset : rowOffset+cols]
		out[rowIndex] = storeMathResult(logSumExpRowFloat32(row), format)
	}
}

func outerReference(left, right, out []float32, format dtype.DType) {
	if format == dtype.Float32 {
		cpumath.OuterGeneric(left, right, out)
		return
	}

	rightLen := len(right)

	for leftIndex, leftValue := range left {
		rowOffset := leftIndex * rightLen

		for rightIndex, rightValue := range right {
			out[rowOffset+rightIndex] = storeMathResult(leftValue*rightValue, format)
		}
	}
}

func logSumExpRowFloat32(row []float32) float32 {
	maximum := row[0]

	for _, candidate := range row[1:] {
		if candidate > maximum {
			maximum = candidate
		}
	}

	var accumulator float32

	for _, candidate := range row {
		accumulator += float32(math.Exp(float64(candidate - maximum)))
	}

	return maximum + float32(math.Log(float64(accumulator)))
}

func storeMathResult(value float32, format dtype.DType) float32 {
	switch format {
	case dtype.Float32:
		return value
	case dtype.Float16:
		return dtype.Fromfloat32(value).Float32()
	case dtype.BFloat16:
		return dtype.NewBfloat16FromFloat32(value).Float32()
	default:
		panic(fmt.Sprintf("unsupported dtype %v", format))
	}
}
