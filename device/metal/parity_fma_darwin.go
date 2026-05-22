//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"context"
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

/*
metalFMAFloat32ForTest evaluates fma(a,b,c) on the same GPU as norm/SwiGLU kernels.
*/
func metalFMAFloat32ForTest(
	testingObject testing.TB,
	backend *Backend,
	a float32,
	b float32,
	c float32,
) float32 {
	testingObject.Helper()

	values := metalFMAFloat32VectorForTest(testingObject, backend, []float32{a}, []float32{b}, []float32{c})

	return values[0]
}

/*
metalFMAFloat32VectorForTest evaluates out[i]=fma(a[i],b[i],c[i]) on GPU.
*/
func metalFMAFloat32VectorForTest(
	testingObject testing.TB,
	backend *Backend,
	aValues []float32,
	bValues []float32,
	cValues []float32,
) []float32 {
	testingObject.Helper()

	if len(aValues) != len(bValues) || len(aValues) != len(cValues) {
		testingObject.Fatalf("fma vector lengths mismatch: %d %d %d", len(aValues), len(bValues), len(cValues))
	}

	if len(aValues) == 0 {
		return nil
	}

	shape := mustShapeForTest(testingObject, []int{len(aValues)})
	aTensor, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(aValues))
	if err != nil {
		testingObject.Fatalf("Upload fma a failed: %v", err)
	}

	defer func() {
		if closeErr := aTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close fma a failed: %v", closeErr)
		}
	}()

	bTensor, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(bValues))
	if err != nil {
		testingObject.Fatalf("Upload fma b failed: %v", err)
	}

	defer func() {
		if closeErr := bTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close fma b failed: %v", closeErr)
		}
	}()

	cTensor, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(cValues))
	if err != nil {
		testingObject.Fatalf("Upload fma c failed: %v", err)
	}

	defer func() {
		if closeErr := cTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close fma c failed: %v", closeErr)
		}
	}()

	outTensor := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)

	defer func() {
		if closeErr := outTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close fma out failed: %v", closeErr)
		}
	}()

	if err := runMetalFMAFloat32(context.Background(), aTensor, bTensor, cTensor, outTensor); err != nil {
		testingObject.Fatalf("runMetalFMAFloat32 failed: %v", err)
	}

	return downloadFloat32ForTest(testingObject, backend, outTensor)
}

/*
applyNorm3DExpectedRowGPU applies fma(centered*invStdDev, scale, bias) via Metal fma_float32.
*/
func applyNorm3DExpectedRowGPU(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	out []float32,
	scale float32,
	bias float32,
	mean float32,
	invStdDev float32,
) {
	testingObject.Helper()

	if len(input) == 0 {
		return
	}

	aValues := make([]float32, len(input))
	bValues := make([]float32, len(input))
	cValues := make([]float32, len(input))

	for index, value := range input {
		aValues[index] = (value - mean) * invStdDev
		bValues[index] = scale
		cValues[index] = bias
	}

	result := metalFMAFloat32VectorForTest(testingObject, backend, aValues, bValues, cValues)
	copy(out, result)
}

func runMetalFMAFloat32(
	ctx context.Context,
	aTensor tensor.Tensor,
	bTensor tensor.Tensor,
	cTensor tensor.Tensor,
	outTensor tensor.Tensor,
) error {
	aMetal, bMetal, cMetal, outMetal, err := requireMetalFMAFloat32Tensors(aTensor, bTensor, cTensor, outTensor)
	if err != nil {
		return err
	}

	if aMetal.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outMetal, aMetal, bMetal, cMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_fma_float32(
		aMetal.bridge.device,
		aMetal.buffer,
		bMetal.buffer,
		cMetal.buffer,
		outMetal.buffer,
		C.uint32_t(aMetal.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal fma_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalFMAFloat32Tensors(
	aTensor tensor.Tensor,
	bTensor tensor.Tensor,
	cTensor tensor.Tensor,
	outTensor tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, *metalTensor, error) {
	aMetal, err := requireMetalTensor(aTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	bMetal, err := requireMetalTensor(bTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	cMetal, err := requireMetalTensor(cTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	outMetal, err := requireMetalTensor(outTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	if aMetal.dtype != dtype.Float32 ||
		bMetal.dtype != dtype.Float32 ||
		cMetal.dtype != dtype.Float32 ||
		outMetal.dtype != dtype.Float32 {
		return nil, nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if !aMetal.shape.Equal(bMetal.shape) ||
		!aMetal.shape.Equal(cMetal.shape) ||
		!aMetal.shape.Equal(outMetal.shape) {
		return nil, nil, nil, nil, tensor.ErrShapeMismatch
	}

	if aMetal.bridge != bMetal.bridge ||
		aMetal.bridge != cMetal.bridge ||
		aMetal.bridge != outMetal.bridge {
		return nil, nil, nil, nil, fmt.Errorf("metal fma_float32: tensors belong to different Metal backends")
	}

	return aMetal, bMetal, cMetal, outMetal, nil
}

/*
metalInvStdDevPreciseFloat32ForTest matches NCS kernels: out[i] = 1/precise::sqrt(values[i]).
*/
func metalInvStdDevPreciseFloat32ForTest(
	testingObject testing.TB,
	backend *Backend,
	variancePlusEpsilon []float32,
) []float32 {
	testingObject.Helper()

	if len(variancePlusEpsilon) == 0 {
		return nil
	}

	shape := mustShapeForTest(testingObject, []int{len(variancePlusEpsilon)})
	inputTensor, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(variancePlusEpsilon))
	if err != nil {
		testingObject.Fatalf("Upload inv_std_dev input failed: %v", err)
	}

	defer func() {
		if closeErr := inputTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close inv_std_dev input failed: %v", closeErr)
		}
	}()

	outTensor := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)

	defer func() {
		if closeErr := outTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close inv_std_dev out failed: %v", closeErr)
		}
	}()

	if err := runMetalInvStdDevFloat32(context.Background(), inputTensor, outTensor); err != nil {
		testingObject.Fatalf("runMetalInvStdDevFloat32 failed: %v", err)
	}

	return downloadFloat32ForTest(testingObject, backend, outTensor)
}

func runMetalInvStdDevFloat32(
	ctx context.Context,
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) error {
	inputMetal, outMetal, err := requireMetalInvStdDevFloat32Tensors(inputTensor, outTensor)
	if err != nil {
		return err
	}

	if inputMetal.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outMetal, inputMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_inv_std_dev_float32(
		inputMetal.bridge.device,
		inputMetal.buffer,
		outMetal.buffer,
		C.uint32_t(inputMetal.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal inv_std_dev_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalInvStdDevFloat32Tensors(
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputMetal, err := requireMetalTensor(inputTensor)
	if err != nil {
		return nil, nil, err
	}

	outMetal, err := requireMetalTensor(outTensor)
	if err != nil {
		return nil, nil, err
	}

	if inputMetal.dtype != dtype.Float32 || outMetal.dtype != dtype.Float32 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if !inputMetal.shape.Equal(outMetal.shape) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if inputMetal.bridge != outMetal.bridge {
		return nil, nil, fmt.Errorf("metal inv_std_dev_float32: tensors belong to different Metal backends")
	}

	return inputMetal, outMetal, nil
}
