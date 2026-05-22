//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalNormConfig struct {
	input          *metalTensor
	scale          *metalTensor
	bias           *metalTensor
	out            *metalTensor
	elementDType   metalElementDType
	rows           uint32
	cols           uint32
	rowsPerBatch   uint32
	modulationCols uint32
}

func runMetalLayerNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalNorm(input, scale, bias, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale, config.bias)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_layernorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal layernorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalRMSNorm(input tensor.Tensor, scale tensor.Tensor, out tensor.Tensor) error {
	return runMetalRMSNormConfigured(input, scale, out, DefaultRMSNormConfig())
}

func runMetalRMSNormConfigured(
	input tensor.Tensor,
	scale tensor.Tensor,
	out tensor.Tensor,
	rmsConfig RMSNormConfig,
) error {
	config, err := requireMetalNorm(input, scale, nil, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	epsilon := rmsConfig.Epsilon
	if epsilon <= 0 {
		epsilon = DefaultRMSNormConfig().Epsilon
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_rmsnorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.float(epsilon),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal rmsnorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func (backend *Backend) AdaptiveRMSNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
) error {
	return runMetalAdaptiveRMSNorm(input, modulation, out)
}

func runMetalAdaptiveRMSNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalAdaptiveNorm(input, modulation, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_adaptive_rmsnorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal adaptive rmsnorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalModulatedLayerNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	config, err := requireMetalModulatedNorm(input, modulation, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_modulated_layernorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint32_t(config.rowsPerBatch),
		C.uint32_t(config.modulationCols),
		C.uint32_t(modulationSet),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal modulated layernorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func (backend *Backend) ModulatedLayerNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	return runMetalModulatedLayerNorm(input, modulation, out, modulationSet)
}

func RunModulatedLayerNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	return runMetalModulatedLayerNorm(input, modulation, out, modulationSet)
}

func runMetalGatedResidual(
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	config, branchTensor, err := requireMetalGatedResidual(residual, branch, modulation, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, branchTensor, config.scale)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_gated_residual(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		branchTensor.buffer,
		config.scale.buffer,
		config.out.buffer,
		C.uint32_t(config.input.shape.Len()),
		C.uint32_t(config.cols),
		C.uint32_t(config.rowsPerBatch),
		C.uint32_t(config.modulationCols),
		C.uint32_t(modulationSet),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal gated residual: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func (backend *Backend) GatedResidual(
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	return runMetalGatedResidual(residual, branch, modulation, out, modulationSet)
}

func RunGatedResidual(
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
	modulationSet int,
) error {
	return runMetalGatedResidual(residual, branch, modulation, out, modulationSet)
}

func (backend *Backend) BatchNormDenorm(
	input tensor.Tensor,
	mean tensor.Tensor,
	variance tensor.Tensor,
	out tensor.Tensor,
) error {
	inputTensor, meanTensor, varianceTensor, outTensor, rows, channels, spatial, err :=
		requireMetalBatchNormDenorm(input, mean, variance, out)
	if err != nil {
		return err
	}

	if outTensor.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor, meanTensor, varianceTensor)
	if err != nil {
		return err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		metalCompletions.Fail(token, err)
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_batchnorm_denorm(
		inputTensor.bridge.device,
		C.int(elementDType),
		inputTensor.buffer,
		meanTensor.buffer,
		varianceTensor.buffer,
		outTensor.buffer,
		C.uint32_t(rows),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal batchnorm denorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	config, err := metalNormTensors(input, scale, bias, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	rows, cols, err := metalNormDims(config.input, config.scale, config.out)
	if err != nil {
		return metalNormConfig{}, err
	}

	if err := requireMetalNormBias(config.bias, cols); err != nil {
		return metalNormConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.rows = uint32(rows)
	config.cols = uint32(cols)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalAdaptiveNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	config, err := metalNormTensors(input, modulation, nil, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	rows, cols, err := metalAdaptiveNormDims(config.input, config.scale, config.out)
	if err != nil {
		return metalNormConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.rows = uint32(rows)
	config.cols = uint32(cols)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalModulatedNorm(
	input tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	config, err := metalNormTensors(input, modulation, nil, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	rows, cols, rowsPerBatch, modulationCols, err := metalModulatedNormDims(
		config.input, config.scale, config.out,
	)
	if err != nil {
		return metalNormConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.rows = uint32(rows)
	config.cols = uint32(cols)
	config.rowsPerBatch = uint32(rowsPerBatch)
	config.modulationCols = uint32(modulationCols)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalGatedResidual(
	residual tensor.Tensor,
	branch tensor.Tensor,
	modulation tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, *metalTensor, error) {
	config, err := requireMetalModulatedNorm(residual, modulation, out)
	if err != nil {
		return metalNormConfig{}, nil, err
	}

	branchTensor, err := requireMetalTensor(branch)
	if err != nil {
		return metalNormConfig{}, nil, err
	}

	if !branchTensor.shape.Equal(config.input.shape) || branchTensor.dtype != config.input.dtype {
		return metalNormConfig{}, nil, tensor.ErrShapeMismatch
	}

	if branchTensor.bridge != config.input.bridge {
		return metalNormConfig{}, nil, errors.New("metal normalization: tensors belong to different Metal backends")
	}

	return config, branchTensor, nil
}

func requireMetalBatchNormDenorm(
	input tensor.Tensor,
	mean tensor.Tensor,
	variance tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, *metalTensor, int, int, int, error) {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	meanTensor, err := requireMetalTensor(mean)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	varianceTensor, err := requireMetalTensor(variance)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	if inputTensor.bridge != meanTensor.bridge || inputTensor.bridge != varianceTensor.bridge ||
		inputTensor.bridge != outTensor.bridge {
		return nil, nil, nil, nil, 0, 0, 0, errors.New("metal normalization: tensors belong to different Metal backends")
	}

	if inputTensor.dtype != meanTensor.dtype || inputTensor.dtype != varianceTensor.dtype ||
		inputTensor.dtype != outTensor.dtype {
		return nil, nil, nil, nil, 0, 0, 0, tensor.ErrDTypeMismatch
	}

	if !inputTensor.shape.Equal(outTensor.shape) {
		return nil, nil, nil, nil, 0, 0, 0, tensor.ErrShapeMismatch
	}

	dims := inputTensor.shape.Dims()
	if len(dims) != 4 {
		return nil, nil, nil, nil, 0, 0, 0, tensor.ErrShapeMismatch
	}

	channels := dims[1]
	spatial := dims[2] * dims[3]
	rows := dims[0] * channels

	if !meanTensor.shape.Equal(vectorShape(channels)) || !varianceTensor.shape.Equal(vectorShape(channels)) {
		return nil, nil, nil, nil, 0, 0, 0, tensor.ErrShapeMismatch
	}

	return inputTensor, meanTensor, varianceTensor, outTensor, rows, channels, spatial, nil
}

func vectorShape(length int) tensor.Shape {
	shape, _ := tensor.NewShape([]int{length})
	return shape
}

func metalNormTensors(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	inputTensor, scaleTensor, outTensor, err := requireMetalNormCoreTensors(input, scale, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	config := metalNormConfig{input: inputTensor, scale: scaleTensor, out: outTensor}
	if bias == nil {
		return config, requireMetalNormSameDevice(config)
	}

	biasTensor, err := requireMetalTensor(bias)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.bias = biasTensor
	return config, requireMetalNormSameDevice(config)
}

func requireMetalNormCoreTensors(
	input tensor.Tensor,
	scale tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, nil, err
	}

	scaleTensor, err := requireMetalTensor(scale)
	if err != nil {
		return nil, nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, err
	}

	return inputTensor, scaleTensor, outTensor, nil
}

func requireMetalNormSameDevice(config metalNormConfig) error {
	if config.input.dtype != config.scale.dtype || config.input.dtype != config.out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.bias != nil && config.bias.dtype != config.input.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.input.bridge != config.scale.bridge || config.input.bridge != config.out.bridge {
		return errors.New("metal normalization: tensors belong to different Metal backends")
	}

	if config.bias != nil && config.bias.bridge != config.input.bridge {
		return errors.New("metal normalization: tensors belong to different Metal backends")
	}

	return nil
}

func metalNormDims(
	input *metalTensor,
	scale *metalTensor,
	out *metalTensor,
) (int, int, error) {
	if !input.shape.Equal(out.shape) {
		fmt.Printf("metalNormDims: shape mismatch: input=%v, out=%v\n", input.shape.Dims(), out.shape.Dims())
		return 0, 0, tensor.ErrShapeMismatch
	}

	dims := input.shape.Dims()
	if len(dims) == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	cols := dims[len(dims)-1]
	if cols == 0 {
		return 0, 0, nil
	}

	scaleDims := scale.shape.Dims()
	if len(scaleDims) != 1 || scaleDims[0] != cols {
		fmt.Printf("metalNormDims: scale mismatch: scale=%v, cols=%d\n", scaleDims, cols)
		return 0, 0, tensor.ErrShapeMismatch
	}

	rows := input.shape.Len() / cols
	if rows > math.MaxUint32 || cols > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, nil
}

func metalAdaptiveNormDims(
	input *metalTensor,
	modulation *metalTensor,
	out *metalTensor,
) (int, int, error) {
	if !input.shape.Equal(out.shape) {
		return 0, 0, tensor.ErrShapeMismatch
	}

	dims := input.shape.Dims()
	if len(dims) < 2 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	cols := dims[len(dims)-1]
	if cols == 0 {
		return 0, 0, nil
	}

	modulationDims := modulation.shape.Dims()
	if len(modulationDims) != 2 || modulationDims[0] != dims[0] || modulationDims[1] != cols*2 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	rows := input.shape.Len() / cols
	if rows > math.MaxUint32 || cols > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, nil
}

func metalModulatedNormDims(
	input *metalTensor,
	modulation *metalTensor,
	out *metalTensor,
) (int, int, int, int, error) {
	if !input.shape.Equal(out.shape) {
		return 0, 0, 0, 0, fmt.Errorf(
			"metal modulated norm: input shape %v != out shape %v: %w",
			input.shape.Dims(),
			out.shape.Dims(),
			tensor.ErrShapeMismatch,
		)
	}

	dims := input.shape.Dims()
	if len(dims) < 2 {
		return 0, 0, 0, 0, fmt.Errorf(
			"metal modulated norm: input rank %d: %w",
			len(dims),
			tensor.ErrShapeMismatch,
		)
	}

	cols := dims[len(dims)-1]
	if cols == 0 {
		return 0, 0, 0, 0, nil
	}

	modulationDims := modulation.shape.Dims()
	if len(modulationDims) != 2 || modulationDims[0] != dims[0] || modulationDims[1]%cols != 0 {
		return 0, 0, 0, 0, fmt.Errorf(
			"metal modulated norm: input shape %v modulation shape %v cols %d: %w",
			dims,
			modulationDims,
			cols,
			tensor.ErrShapeMismatch,
		)
	}

	if modulationDims[1] != cols*3 && modulationDims[1] != cols*6 {
		return 0, 0, 0, 0, fmt.Errorf(
			"metal modulated norm: modulation cols %d incompatible with hidden cols %d: %w",
			modulationDims[1],
			cols,
			tensor.ErrShapeMismatch,
		)
	}

	rows := input.shape.Len() / cols
	rowsPerBatch := rows / dims[0]
	if rows > math.MaxUint32 || cols > math.MaxUint32 || rowsPerBatch > math.MaxUint32 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, rowsPerBatch, modulationDims[1], nil
}

func requireMetalNormBias(bias *metalTensor, cols int) error {
	if bias == nil {
		return nil
	}

	biasDims := bias.shape.Dims()
	if len(biasDims) != 1 || biasDims[0] != cols {
		return tensor.ErrShapeMismatch
	}

	return nil
}
