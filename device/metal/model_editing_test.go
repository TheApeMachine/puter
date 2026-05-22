package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var modelEditingStorageDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func TestKernelRegistry_MetalModelEditingKernels(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range modelEditingStorageDTypes {
		storageDType := storageDType

		for _, elementCount := range parityElementCounts {
			elementCount := elementCount

			testingObject.Run(
				testNameForStorageDTypeAndElementCount(storageDType, elementCount),
				func(testingObject *testing.T) {
					convey.Convey("Given Metal tensors for model-editing kernels", testingObject, func() {
						runWeightGraftAddParityCase(testingObject, backend, elementCount, storageDType)
					})
				},
			)
		}
	}
}

func runWeightGraftAddParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	storageDType dtype.DType,
) {
	fixture := weightGraftFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	weights := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.weightsBytes)
	injection := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.injectionBytes)
	defer closeBenchmarkTensors(weights, injection)

	err := lookupWeightGraftAddKernel(testingObject, storageDType).Run(weights, injection)
	convey.So(err, convey.ShouldBeNil)
	assertUtilityStorageForTest(testingObject, backend, weights, storageDType, fixture.expectedBytes)
}

func lookupWeightGraftAddKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("weight_graft_add", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("weight_graft_add Metal kernel not registered for %v", storageDType)
	}

	return kernel
}
