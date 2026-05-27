package layernorm

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const rmsNormConfigMaxULP = 0

func TestRMSNormFloat32ConfigEpsilonParity(testingT *testing.T) {
	convey.Convey("Given RMSNorm float32 receives epsilon from config", testingT, func() {
		config := device.RMSNormConfig{Epsilon: 1e-5}

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match reference for N=%d", length), func() {
				rows := 2
				input := rmsNormInput(rows * length)
				scale := rmsNormScale(length)
				output := make([]float32, rows*length)
				expected := rmsNormReferenceFloat32(input, scale, rows, length, config)

				Default.RMSNorm(
					config,
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&scale[0]),
					unsafe.Pointer(&output[0]),
					rows,
					length,
					dtype.Float32,
				)

				parity.AssertFloat32SlicesWithinULP(
					testingT,
					output,
					expected,
					rmsNormConfigMaxULP,
				)
			})
		}
	})
}

func BenchmarkRMSNormFloat32ConfigEpsilon(benchmark *testing.B) {
	config := device.RMSNormConfig{Epsilon: 1e-5}
	rows := 16
	lastDim := 8192
	input := rmsNormInput(rows * lastDim)
	scale := rmsNormScale(lastDim)
	output := make([]float32, rows*lastDim)

	benchmark.ReportAllocs()

	for benchmark.Loop() {
		Default.RMSNorm(
			config,
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&scale[0]),
			unsafe.Pointer(&output[0]),
			rows,
			lastDim,
			dtype.Float32,
		)
	}
}

func rmsNormInput(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32(index%67-33) / 19
	}

	return values
}

func rmsNormScale(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = 1 + float32(index%31-15)/128
	}

	return values
}

func rmsNormReferenceFloat32(
	input []float32,
	scale []float32,
	rows int,
	lastDim int,
	config device.RMSNormConfig,
) []float32 {
	output := make([]float32, len(input))

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * lastDim
		row := input[rowOffset : rowOffset+lastDim]
		outputRow := output[rowOffset : rowOffset+lastDim]
		sumOfSquares := DotFloat32Native(row, row)
		meanSquare := float64(sumOfSquares) / float64(lastDim)
		invRMS := float32(1 / math.Sqrt(meanSquare+config.Epsilon))

		for columnIndex := range scale {
			outputRow[columnIndex] = row[columnIndex] * (invRMS * scale[columnIndex])
		}
	}

	return output
}
