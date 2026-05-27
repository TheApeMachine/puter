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

const adaptiveRMSNormMaxULP = 2

func TestAdaptiveRMSNormFloat32Parity(testingObject *testing.T) {
	convey.Convey("Given AdaptiveRMSNorm float32 inputs", testingObject, func() {
		config := device.RMSNormConfig{Epsilon: 1e-6}

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match reference for N=%d", length), func() {
				rowsPerBatch := 3
				batches := 2
				rows := batches * rowsPerBatch
				modulationCols := 2 * length
				input := modulatedLayerNormInput(rows * length)
				modulation := modulatedLayerNormModulation(batches * modulationCols)
				output := make([]float32, rows*length)
				expected := adaptiveRMSNormReferenceFloat32(
					config,
					input,
					modulation,
					rows,
					length,
					rowsPerBatch,
					modulationCols,
				)

				Default.AdaptiveRMSNorm(
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
					adaptiveRMSNormMaxULP,
				)
			})
		}
	})
}

func BenchmarkAdaptiveRMSNormFloat32(benchmark *testing.B) {
	config := device.RMSNormConfig{Epsilon: 1e-6}
	rowsPerBatch := 16
	batches := 2
	rows := batches * rowsPerBatch
	lastDim := 8192
	modulationCols := 2 * lastDim
	input := modulatedLayerNormInput(rows * lastDim)
	modulation := modulatedLayerNormModulation(batches * modulationCols)
	output := make([]float32, rows*lastDim)

	benchmark.ReportAllocs()

	for benchmark.Loop() {
		Default.AdaptiveRMSNorm(
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

func adaptiveRMSNormReferenceFloat32(
	config device.RMSNormConfig,
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
		batchIndex := rowIndex / rowsPerBatch
		modulationOffset := batchIndex * modulationCols
		row := input[rowOffset : rowOffset+lastDim]
		outRow := output[rowOffset : rowOffset+lastDim]

		adaptiveRMSNormReferenceRow(config, row, modulation[modulationOffset:], outRow)
	}

	return output
}

func adaptiveRMSNormReferenceRow(
	config device.RMSNormConfig,
	row []float32,
	modulation []float32,
	output []float32,
) {
	invRMS := float32(1.0 / math.Sqrt(float64(adaptiveRMSNormMeanSquareF32(row))+config.Epsilon))

	for columnIndex := range row {
		normalized := row[columnIndex] * invRMS
		scale := modulation[columnIndex]
		shift := modulation[len(row)+columnIndex]
		output[columnIndex] = normalized*(1+scale) + shift
	}
}
