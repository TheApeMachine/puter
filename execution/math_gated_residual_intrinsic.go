package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type gatedResidualDevice interface {
	GatedResidual(
		residual, branch, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols, set int,
		format dtype.DType,
	)
}

type gatedResidualLayout struct {
	dimensions     []int
	rows           int
	lastDim        int
	rowsPerBatch   int
	modulationCols int
	set            int
}

func runGatedResidualIntrinsic(resolver *bindResolver) (any, error) {
	residual, branch, modulation, layout, err := resolver.resolveGatedResidualInputs()

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if residual.Location() == tensor.Host {
		if branch.Location() != tensor.Host || modulation.Location() != tensor.Host || output.Location() != tensor.Host {
			return nil, fmt.Errorf("math.gated_residual host tensor locations do not match")
		}

		return output, copyGatedResidualHost(output, residual, branch, modulation, layout)
	}

	if residual.Location() != branch.Location() || residual.Location() != modulation.Location() ||
		residual.Location() != output.Location() {
		return nil, fmt.Errorf("math.gated_residual tensor locations do not match")
	}

	return output, runGatedResidualDeviceIntrinsic(resolver, output, residual, branch, modulation, layout)
}

func (resolver *bindResolver) resolveGatedResidualInputs() (
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	gatedResidualLayout,
	error,
) {
	residual, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, nil, nil, gatedResidualLayout{}, err
	}

	branch, err := resolver.resolveInputTensor("1")

	if err != nil {
		return nil, nil, nil, gatedResidualLayout{}, err
	}

	modulation, err := resolver.resolveInputTensor("2")

	if err != nil {
		return nil, nil, nil, gatedResidualLayout{}, err
	}

	layout, err := resolver.resolveGatedResidualLayout(residual, branch, modulation)

	if err != nil {
		return nil, nil, nil, gatedResidualLayout{}, err
	}

	return residual, branch, modulation, layout, nil
}

func (resolver *bindResolver) resolveGatedResidualLayout(
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
) (gatedResidualLayout, error) {
	if residual.DType() != branch.DType() || residual.DType() != modulation.DType() {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual dtype mismatch")
	}

	dimensions, err := resolver.resolveInputDimensions("0", residual)

	if err != nil {
		return gatedResidualLayout{}, err
	}

	branchDimensions, err := resolver.resolveInputDimensions("1", branch)

	if err != nil {
		return gatedResidualLayout{}, err
	}

	if !sameDimensions(dimensions, branchDimensions) {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual residual and branch shape mismatch")
	}

	if len(dimensions) < 2 {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual input rank must be >= 2")
	}

	modulationDimensions := modulation.Shape().Dims()

	if len(modulationDimensions) == 0 {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual modulation rank must be >= 1")
	}

	layout := gatedResidualLayout{
		dimensions:     dimensions,
		rows:           productInts(dimensions[:len(dimensions)-1]),
		lastDim:        dimensions[len(dimensions)-1],
		rowsPerBatch:   productInts(dimensions[1 : len(dimensions)-1]),
		modulationCols: modulationDimensions[len(modulationDimensions)-1],
		set:            configInt(resolver.node, "set", resolver.defaultConfigInt("set")),
	}

	if layout.rowsPerBatch <= 0 || layout.rows%layout.rowsPerBatch != 0 {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual invalid rows_per_batch")
	}

	if layout.modulationCols < (layout.set*3+3)*layout.lastDim {
		return gatedResidualLayout{}, fmt.Errorf("math.gated_residual modulation width too small")
	}

	return layout, nil
}

func sameDimensions(left []int, right []int) bool {
	if len(left) != len(right) {
		return false
	}

	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}

	return true
}

func copyGatedResidualHost(
	output tensor.Tensor,
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	layout gatedResidualLayout,
) error {
	if residual.DType() != dtype.Float32 {
		return fmt.Errorf("math.gated_residual host dtype %s is not supported", residual.DType())
	}

	residualPointer, _, err := pointerOf(residual)

	if err != nil {
		return err
	}

	branchPointer, _, err := pointerOf(branch)

	if err != nil {
		return err
	}

	modulationPointer, _, err := pointerOf(modulation)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	copyGatedResidualFloat32(
		unsafe.Slice((*float32)(outputPointer), output.Len()),
		unsafe.Slice((*float32)(residualPointer), residual.Len()),
		unsafe.Slice((*float32)(branchPointer), branch.Len()),
		unsafe.Slice((*float32)(modulationPointer), modulation.Len()),
		layout,
	)

	return nil
}

func copyGatedResidualFloat32(
	output []float32,
	residual []float32,
	branch []float32,
	modulation []float32,
	layout gatedResidualLayout,
) {
	for row := range layout.rows {
		batch := row / layout.rowsPerBatch
		rowOffset := row * layout.lastDim
		modulationOffset := batch*layout.modulationCols + layout.set*layout.lastDim*3 + layout.lastDim*2

		for col := range layout.lastDim {
			index := rowOffset + col
			output[index] = residual[index] + modulation[modulationOffset+col]*branch[index]
		}
	}
}

func runGatedResidualDeviceIntrinsic(
	resolver *bindResolver,
	output tensor.Tensor,
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	layout gatedResidualLayout,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(gatedResidualDevice)

	if !ok {
		return fmt.Errorf(
			"math.gated_residual: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			residual.Location(),
		)
	}

	residualPointer, _, err := pointerOf(residual)

	if err != nil {
		return err
	}

	branchPointer, _, err := pointerOf(branch)

	if err != nil {
		return err
	}

	modulationPointer, _, err := pointerOf(modulation)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	deviceBackend.GatedResidual(
		residualPointer,
		branchPointer,
		modulationPointer,
		outputPointer,
		layout.rows,
		layout.lastDim,
		layout.rowsPerBatch,
		layout.modulationCols,
		layout.set,
		residual.DType(),
	)

	return nil
}
