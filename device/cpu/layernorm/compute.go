package layernorm

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

const layerNormEpsilon = 1e-5

func dispatchLayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		runLayerNormF32(input, scale, bias, output, rows, lastDim)
	case dtype.BFloat16:
		runLayerNormBF16(input, scale, bias, output, rows, lastDim)
	case dtype.Float16:
		runLayerNormF16(input, scale, bias, output, rows, lastDim)
	default:
		panic("layernorm: unsupported dtype")
	}
}

func dispatchRMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	if err := config.Validate(); err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32:
		runRMSNormF32(config, input, scale, output, rows, lastDim)
	case dtype.BFloat16:
		runRMSNormBF16(config, input, scale, output, rows, lastDim)
	case dtype.Float16:
		runRMSNormF16(config, input, scale, output, rows, lastDim)
	default:
		panic("layernorm: unsupported dtype")
	}
}

func dispatchModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	validateModulatedLayerNorm(config, rows, lastDim, rowsPerBatch, modulationCols)

	switch format {
	case dtype.Float32:
		runModulatedLayerNormF32(config, input, modulation, output, rows, lastDim, rowsPerBatch, modulationCols)
	case dtype.BFloat16:
		runModulatedLayerNormBF16(config, input, modulation, output, rows, lastDim, rowsPerBatch, modulationCols)
	case dtype.Float16:
		runModulatedLayerNormF16(config, input, modulation, output, rows, lastDim, rowsPerBatch, modulationCols)
	default:
		panic("modulated layernorm: unsupported dtype")
	}
}

func runLayerNormF32(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*float32)(input), rows*lastDim)
	scaleView := unsafe.Slice((*float32)(scale), lastDim)
	biasView := unsafe.Slice((*float32)(bias), lastDim)
	outputView := unsafe.Slice((*float32)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]

		mean := computeRowMean(row)
		variance := computeRowVariance(row, mean)
		invStdDev := 1.0 / math.Sqrt(variance+layerNormEpsilon)
		applyRowNormalization(row, outRow, scaleView, biasView, mean, invStdDev)
	}
}

func runRMSNormF32(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*float32)(input), rows*lastDim)
	scaleView := unsafe.Slice((*float32)(scale), lastDim)
	outputView := unsafe.Slice((*float32)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRowF32(config, row, outRow, scaleView)
	}
}

func runLayerNormBF16(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), rows*lastDim)
	scaleView := unsafe.Slice((*dtype.BF16)(scale), lastDim)
	biasView := unsafe.Slice((*dtype.BF16)(bias), lastDim)
	outputView := unsafe.Slice((*dtype.BF16)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyLayerNormRowBF16(row, outRow, scaleView, biasView)
	}
}

func runLayerNormF16(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), rows*lastDim)
	scaleView := unsafe.Slice((*dtype.F16)(scale), lastDim)
	biasView := unsafe.Slice((*dtype.F16)(bias), lastDim)
	outputView := unsafe.Slice((*dtype.F16)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyLayerNormRowF16(row, outRow, scaleView, biasView)
	}
}

func runRMSNormBF16(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), rows*lastDim)
	scaleView := unsafe.Slice((*dtype.BF16)(scale), lastDim)
	outputView := unsafe.Slice((*dtype.BF16)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRowBF16(config, row, outRow, scaleView)
	}
}

func runRMSNormF16(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), rows*lastDim)
	scaleView := unsafe.Slice((*dtype.F16)(scale), lastDim)
	outputView := unsafe.Slice((*dtype.F16)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRowF16(config, row, outRow, scaleView)
	}
}

func runModulatedLayerNormF32(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
) {
	inputView := unsafe.Slice((*float32)(input), rows*lastDim)
	modulationView := unsafe.Slice((*float32)(modulation), batchCount(rows, rowsPerBatch)*modulationCols)
	outputView := unsafe.Slice((*float32)(output), rows*lastDim)

	for rowIndex := range rows {
		rowOffset := rowIndex * lastDim
		row := inputView[rowOffset : rowOffset+lastDim]
		outRow := outputView[rowOffset : rowOffset+lastDim]
		modulationOffset := modulatedLayerNormOffset(config, rowIndex, rowsPerBatch, modulationCols, lastDim)
		applyModulatedLayerNormRowF32(
			config,
			row,
			modulationView[modulationOffset:],
			outRow,
		)
	}
}

func runModulatedLayerNormBF16(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), rows*lastDim)
	modulationView := unsafe.Slice((*dtype.BF16)(modulation), batchCount(rows, rowsPerBatch)*modulationCols)
	outputView := unsafe.Slice((*dtype.BF16)(output), rows*lastDim)

	for rowIndex := range rows {
		rowOffset := rowIndex * lastDim
		row := inputView[rowOffset : rowOffset+lastDim]
		outRow := outputView[rowOffset : rowOffset+lastDim]
		modulationOffset := modulatedLayerNormOffset(config, rowIndex, rowsPerBatch, modulationCols, lastDim)
		applyModulatedLayerNormRowBF16(
			config,
			row,
			modulationView[modulationOffset:],
			outRow,
		)
	}
}

func runModulatedLayerNormF16(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), rows*lastDim)
	modulationView := unsafe.Slice((*dtype.F16)(modulation), batchCount(rows, rowsPerBatch)*modulationCols)
	outputView := unsafe.Slice((*dtype.F16)(output), rows*lastDim)

	for rowIndex := range rows {
		rowOffset := rowIndex * lastDim
		row := inputView[rowOffset : rowOffset+lastDim]
		outRow := outputView[rowOffset : rowOffset+lastDim]
		modulationOffset := modulatedLayerNormOffset(config, rowIndex, rowsPerBatch, modulationCols, lastDim)
		applyModulatedLayerNormRowF16(
			config,
			row,
			modulationView[modulationOffset:],
			outRow,
		)
	}
}

func computeRowMean(row []float32) float64 {
	return float64(SumFloat32Native(row)) / float64(len(row))
}

func computeRowVariance(row []float32, mean float64) float64 {
	return float64(LayerNormSquaredDiffSumNative(row, float32(mean))) / float64(len(row))
}

func applyRowNormalization(
	row, outRow, scale, bias []float32,
	mean, invStdDev float64,
) {
	LayerNormApplyRowNative(outRow, row, scale, bias, float32(mean), float32(invStdDev))
}

func applyRMSRowF32(config device.RMSNormConfig, row, outRow, scale []float32) {
	sumOfSquares := DotFloat32Native(row, row)
	meanSquare := float64(sumOfSquares) / float64(len(row))
	invRMS := 1.0 / math.Sqrt(meanSquare+config.Epsilon)
	invRMSf32 := float32(invRMS)

	combined := make([]float32, len(row))

	for index := range scale {
		combined[index] = invRMSf32 * scale[index]
	}

	MulFloat32Native(outRow, row, combined)
}

func applyRMSRowBF16(config device.RMSNormConfig, row, outRow, scale []dtype.BF16) {
	sumOfSquares := DotBFloat16Native(row, row)
	meanSquare := (&sumOfSquares).Float32() / float32(len(row))
	invRMS := float32(1.0 / math.Sqrt(float64(meanSquare)+config.Epsilon))
	combined := make([]dtype.BF16, len(row))

	for index := range scale {
		combined[index] = dtype.NewBfloat16FromFloat32(invRMS * (&scale[index]).Float32())
	}

	MulBFloat16Native(outRow, row, combined)
}

func applyRMSRowF16(config device.RMSNormConfig, row, outRow, scale []dtype.F16) {
	sumOfSquares := DotFloat16Native(row, row)
	meanSquare := sumOfSquares.Float32() / float32(len(row))
	invRMS := float32(1.0 / math.Sqrt(float64(meanSquare)+config.Epsilon))
	combined := make([]dtype.F16, len(row))

	for index := range scale {
		combined[index] = dtype.Fromfloat32(invRMS * scale[index].Float32())
	}

	MulFloat16Native(outRow, row, combined)
}

func applyLayerNormRowBF16(
	row, outRow, scale, bias []dtype.BF16,
) {
	meanValue := SumBFloat16Native(row)
	mean := (&meanValue).Float32() / float32(len(row))
	variance := layerNormVarianceBF16(row, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance+layerNormEpsilon)))

	for index := range row {
		normalized := ((&row[index]).Float32() - mean) * invStdDev
		out := normalized*(&scale[index]).Float32() + (&bias[index]).Float32()
		outRow[index] = dtype.NewBfloat16FromFloat32(out)
	}
}

func applyLayerNormRowF16(
	row, outRow, scale, bias []dtype.F16,
) {
	meanValue := SumFloat16Native(row)
	mean := meanValue.Float32() / float32(len(row))
	variance := layerNormVarianceF16(row, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance+layerNormEpsilon)))

	for index := range row {
		normalized := (row[index].Float32() - mean) * invStdDev
		out := normalized*scale[index].Float32() + bias[index].Float32()
		outRow[index] = dtype.Fromfloat32(out)
	}
}

func applyModulatedLayerNormRowF32(
	config device.ModulatedLayerNormConfig,
	row, modulation, outRow []float32,
) {
	mean := computeRowMean(row)
	variance := computeRowVariance(row, mean)
	invStdDev := float32(1.0 / math.Sqrt(variance+config.Epsilon))
	applyModulatedLayerNormValuesF32(row, modulation, outRow, float32(mean), invStdDev)
}

func applyModulatedLayerNormValuesF32(
	row, modulation, outRow []float32,
	mean, invStdDev float32,
) {
	cols := len(row)

	for columnIndex := range row {
		normalized := (row[columnIndex] - mean) * invStdDev
		shift := modulation[columnIndex]
		scale := modulation[cols+columnIndex]
		outRow[columnIndex] = normalized*(1+scale) + shift
	}
}

func applyModulatedLayerNormRowBF16(
	config device.ModulatedLayerNormConfig,
	row, modulation, outRow []dtype.BF16,
) {
	meanValue := SumBFloat16Native(row)
	mean := (&meanValue).Float32() / float32(len(row))
	variance := layerNormVarianceBF16(row, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance)+config.Epsilon))
	cols := len(row)

	for columnIndex := range row {
		normalized := ((&row[columnIndex]).Float32() - mean) * invStdDev
		shift := (&modulation[columnIndex]).Float32()
		scale := (&modulation[cols+columnIndex]).Float32()
		outRow[columnIndex] = dtype.NewBfloat16FromFloat32(normalized*(1+scale) + shift)
	}
}

func applyModulatedLayerNormRowF16(
	config device.ModulatedLayerNormConfig,
	row, modulation, outRow []dtype.F16,
) {
	meanValue := SumFloat16Native(row)
	mean := meanValue.Float32() / float32(len(row))
	variance := layerNormVarianceF16(row, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance)+config.Epsilon))
	cols := len(row)

	for columnIndex := range row {
		normalized := (row[columnIndex].Float32() - mean) * invStdDev
		shift := modulation[columnIndex].Float32()
		scale := modulation[cols+columnIndex].Float32()
		outRow[columnIndex] = dtype.Fromfloat32(normalized*(1+scale) + shift)
	}
}

func layerNormVarianceBF16(row []dtype.BF16, mean float32) float32 {
	var variance float32

	for index := range row {
		delta := (&row[index]).Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}

func layerNormVarianceF16(row []dtype.F16, mean float32) float32 {
	var variance float32

	for index := range row {
		delta := row[index].Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}

func validateModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	rows, lastDim, rowsPerBatch, modulationCols int,
) {
	if err := config.Validate(); err != nil {
		panic(err)
	}

	if rowsPerBatch <= 0 {
		panic("modulated layernorm: rowsPerBatch must be positive")
	}

	if rows%rowsPerBatch != 0 {
		panic("modulated layernorm: rows must be divisible by rowsPerBatch")
	}

	if modulationCols < requiredModulationCols(config, lastDim) {
		panic("modulated layernorm: modulation width too small")
	}
}

func requiredModulationCols(config device.ModulatedLayerNormConfig, lastDim int) int {
	return (config.Set*3 + 2) * lastDim
}

func batchCount(rows, rowsPerBatch int) int {
	return rows / rowsPerBatch
}

func modulatedLayerNormOffset(
	config device.ModulatedLayerNormConfig,
	rowIndex, rowsPerBatch, modulationCols, lastDim int,
) int {
	batchIndex := rowIndex / rowsPerBatch
	return batchIndex*modulationCols + config.Set*lastDim*3
}
