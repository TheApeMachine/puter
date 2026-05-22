//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

/*
metalFMAFloat32Scalar evaluates fma(a,b,c) on the same GPU as norm/SwiGLU kernels.
*/
func metalFMAFloat32Scalar(
	testingObject testing.TB,
	backend *Backend,
	a float32,
	b float32,
	c float32,
) float32 {
	testingObject.Helper()

	values := metalFMAFloat32Vector(testingObject, backend, []float32{a}, []float32{b}, []float32{c})

	return values[0]
}

/*
metalFMAFloat32Vector evaluates out[i]=fma(a[i],b[i],c[i]) on GPU.
*/
func metalFMAFloat32Vector(
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

func metalUnaryNamedFloat32Vector(
	testingObject testing.TB,
	backend *Backend,
	kernelName string,
	inputValues []float32,
) []float32 {
	testingObject.Helper()

	if len(inputValues) == 0 {
		return nil
	}

	shape := mustShapeForTest(testingObject, []int{len(inputValues)})
	inputTensor, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(inputValues))
	if err != nil {
		testingObject.Fatalf("Upload %s input failed: %v", kernelName, err)
	}

	defer func() {
		if closeErr := inputTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close %s input failed: %v", kernelName, closeErr)
		}
	}()

	outTensor := emptyTensorForTest(testingObject, backend, shape, dtype.Float32)

	defer func() {
		if closeErr := outTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close %s out failed: %v", kernelName, closeErr)
		}
	}()

	if err := runMetalUnaryNamedFloat32(context.Background(), kernelName, inputTensor, outTensor); err != nil {
		testingObject.Fatalf("runMetalUnaryNamedFloat32 %s failed: %v", kernelName, err)
	}

	return downloadFloat32ForTest(testingObject, backend, outTensor)
}

func metalSiluFloat32Vector(
	testingObject testing.TB,
	backend *Backend,
	inputValues []float32,
) []float32 {
	return metalUnaryNamedFloat32Vector(testingObject, backend, "swiglu_silu_float32", inputValues)
}

func metalHawkesExpFloat32Vector(
	testingObject testing.TB,
	backend *Backend,
	inputValues []float32,
) []float32 {
	return metalUnaryNamedFloat32Vector(testingObject, backend, "hawkes_exp_float32", inputValues)
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

	result := metalFMAFloat32Vector(testingObject, backend, aValues, bValues, cValues)
	copy(out, result)
}

/*
applyNorm3DAffineSliceGPU applies fma((x-mean)*invStdDev, scale, bias) on a contiguous slice with per-element scale/bias.
*/
func applyNorm3DAffineSliceGPU(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	out []float32,
	scaleByElement []float32,
	biasByElement []float32,
	mean float32,
	invStdDev float32,
) {
	testingObject.Helper()

	if len(input) == 0 {
		return
	}

	if len(scaleByElement) != len(input) || len(biasByElement) != len(input) {
		testingObject.Fatalf(
			"norm affine slice lengths mismatch: input=%d scale=%d bias=%d",
			len(input), len(scaleByElement), len(biasByElement),
		)
	}

	aValues := make([]float32, len(input))
	for index, value := range input {
		aValues[index] = (value - mean) * invStdDev
	}

	result := metalFMAFloat32Vector(testingObject, backend, aValues, scaleByElement, biasByElement)
	copy(out, result)
}

/*
metalInvStdDevPreciseFloat32 matches NCS kernels: out[i] = 1/precise::sqrt(values[i]).
*/
func metalInvStdDevPreciseFloat32(
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
