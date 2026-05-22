//go:build darwin && cgo

package metal

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

/*
hawkesIntensityMetalExpected runs hawkes_intensity_float32 on GPU for test gold values.
*/
func hawkesIntensityMetalExpected(
	testingObject testing.TB,
	backend *Backend,
	events []float32,
	queries []float32,
	mu float32,
	alpha float32,
	beta float32,
) []float32 {
	testingObject.Helper()

	eventCount := len(events)
	fixture := hawkesIntensityFixture{
		eventBytes:    dtypeconvert.Float32ToBytes(events),
		queryBytes:    dtypeconvert.Float32ToBytes(queries),
		baselineBytes: dtypeconvert.Float32ToBytes([]float32{mu}),
		alphaBytes:    dtypeconvert.Float32ToBytes([]float32{alpha}),
		betaBytes:     dtypeconvert.Float32ToBytes([]float32{beta}),
	}
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{len(queries)})
	eventsTensor, queryTensor, baseline, alphaTensor, betaTensor, out := hawkesIntensityTensorsForTest(
		testingObject, backend, dtype.Float32, eventShape, outShape, fixture,
	)

	defer closeBenchmarkTensors(eventsTensor, queryTensor, baseline, alphaTensor, betaTensor, out)

	err := lookupHawkesIntensityKernel(testingObject, dtype.Float32).Run(
		eventsTensor, queryTensor, baseline, alphaTensor, betaTensor, out,
	)
	if err != nil {
		testingObject.Fatalf("hawkes_intensity gold Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		testingObject.Fatalf("hawkes_intensity gold Download failed: %v", err)
	}

	return decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
}

/*
hawkesKernelMatrixMetalExpected runs hawkes_kernel_matrix_float32 on GPU for test gold values.
*/
func hawkesKernelMatrixMetalExpected(
	testingObject testing.TB,
	backend *Backend,
	events []float32,
	alpha float32,
	beta float32,
) []float32 {
	testingObject.Helper()

	eventCount := len(events)
	fixture := hawkesKernelMatrixFixture{
		eventBytes: dtypeconvert.Float32ToBytes(events),
		alphaBytes: dtypeconvert.Float32ToBytes([]float32{alpha}),
		betaBytes:  dtypeconvert.Float32ToBytes([]float32{beta}),
	}
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{eventCount, eventCount})
	eventsTensor, alphaTensor, betaTensor, out := hawkesKernelMatrixTensorsForTest(
		testingObject, backend, dtype.Float32, eventShape, outShape, fixture,
	)

	defer closeBenchmarkTensors(eventsTensor, alphaTensor, betaTensor, out)

	err := lookupHawkesKernelMatrixKernel(testingObject, dtype.Float32).Run(
		eventsTensor, alphaTensor, betaTensor, out,
	)
	if err != nil {
		testingObject.Fatalf("hawkes_kernel_matrix gold Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		testingObject.Fatalf("hawkes_kernel_matrix gold Download failed: %v", err)
	}

	return decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
}
