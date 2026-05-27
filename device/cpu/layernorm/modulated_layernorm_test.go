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

const modulatedLayerNormMaxULP = 0

func TestModulatedLayerNormFloat32Parity(testingObject *testing.T) {
	convey.Convey("Given ModulatedLayerNorm float32 inputs", testingObject, func() {
		config := device.ModulatedLayerNormConfig{Epsilon: 1e-6, Set: 1}

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match reference for N=%d", length), func() {
				rowsPerBatch := 3
				batches := 2
				rows := batches * rowsPerBatch
				modulationCols := 6 * length
				input := modulatedLayerNormInput(rows * length)
				modulation := modulatedLayerNormModulation(batches * modulationCols)
				output := make([]float32, rows*length)
				expected := modulatedLayerNormReferenceFloat32(
					config,
					input,
					modulation,
					rows,
					length,
					rowsPerBatch,
					modulationCols,
				)

				Default.ModulatedLayerNorm(
					config,
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&modulation[0]),
					unsafe.Pointer(&output[0]),
					rows,
					length,
					rowsPerBatch,
					modulationCols,
					dtype.Float32,
				)

				parity.AssertFloat32SlicesWithinULP(
					testingObject,
					output,
					expected,
					modulatedLayerNormMaxULP,
				)
			})
		}
	})
}

func BenchmarkModulatedLayerNormFloat32(benchmark *testing.B) {
	config := device.ModulatedLayerNormConfig{Epsilon: 1e-6, Set: 1}
	rowsPerBatch := 16
	batches := 2
	rows := batches * rowsPerBatch
	lastDim := 8192
	modulationCols := 6 * lastDim
	input := modulatedLayerNormInput(rows * lastDim)
	modulation := modulatedLayerNormModulation(batches * modulationCols)
	output := make([]float32, rows*lastDim)

	benchmark.ReportAllocs()

	for benchmark.Loop() {
		Default.ModulatedLayerNorm(
			config,
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&modulation[0]),
			unsafe.Pointer(&output[0]),
			rows,
			lastDim,
			rowsPerBatch,
			modulationCols,
			dtype.Float32,
		)
	}
}

func modulatedLayerNormInput(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32(index%97-48) / 23
	}

	return values
}

func modulatedLayerNormModulation(count int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32(index%37-18) / 211
	}

	return values
}

func modulatedLayerNormReferenceFloat32(
	config device.ModulatedLayerNormConfig,
	input []float32,
	modulation []float32,
	rows int,
	lastDim int,
	rowsPerBatch int,
	modulationCols int,
) []float32 {
	output := make([]float32, len(input))

	for rowIndex := range rows {
		rowOffset := rowIndex * lastDim
		row := input[rowOffset : rowOffset+lastDim]
		outRow := output[rowOffset : rowOffset+lastDim]
		modulationOffset := modulatedLayerNormReferenceOffset(
			config,
			rowIndex,
			rowsPerBatch,
			modulationCols,
			lastDim,
		)

		modulatedLayerNormReferenceRow(
			config,
			row,
			modulation[modulationOffset:],
			outRow,
		)
	}

	return output
}

func modulatedLayerNormReferenceOffset(
	config device.ModulatedLayerNormConfig,
	rowIndex int,
	rowsPerBatch int,
	modulationCols int,
	lastDim int,
) int {
	batchIndex := rowIndex / rowsPerBatch
	return batchIndex*modulationCols + config.Set*lastDim*3
}

func modulatedLayerNormReferenceRow(
	config device.ModulatedLayerNormConfig,
	row []float32,
	modulation []float32,
	output []float32,
) {
	mean := float32(float64(SumFloat32Native(row)) / float64(len(row)))
	variance := float32(
		float64(LayerNormSquaredDiffSumNative(row, mean)) / float64(len(row)),
	)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance)+config.Epsilon))

	for columnIndex := range row {
		normalized := (row[columnIndex] - mean) * invStdDev
		shift := modulation[columnIndex]
		scale := modulation[len(row)+columnIndex]
		output[columnIndex] = normalized*(1+scale) + shift
	}
}
