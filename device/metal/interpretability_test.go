package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var interpretabilityStorageDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func TestKernelRegistry_MetalInterpretabilityKernels(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range interpretabilityStorageDTypes {
		storageDType := storageDType

		for _, elementCount := range parityElementCounts {
			elementCount := elementCount

			testingObject.Run(
				testNameForStorageDTypeAndElementCount(storageDType, elementCount),
				func(testingObject *testing.T) {
					convey.Convey("Given Metal tensors for interpretability kernels", testingObject, func() {
						runActivationSteerParityCase(testingObject, backend, elementCount, storageDType)
					})
				},
			)
		}
	}
}

func runActivationSteerParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	storageDType dtype.DType,
) {
	fixture := activationSteerFixtureForTest(elementCount, storageDType)
	vectorShape := mustShapeForTest(testingObject, []int{elementCount})
	coefficientShape := mustShapeForTest(testingObject, []int{1})
	base := uploadDTypeTensorForTest(testingObject, backend, vectorShape, storageDType, fixture.baseBytes)
	direction := uploadDTypeTensorForTest(testingObject, backend, vectorShape, storageDType, fixture.directionBytes)
	coefficient := uploadDTypeTensorForTest(
		testingObject, backend, coefficientShape, dtype.Float32, fixture.coefficientBytes,
	)
	out := emptyTensorForTest(testingObject, backend, vectorShape, storageDType)
	defer closeBenchmarkTensors(base, direction, coefficient, out)

	err := lookupActivationSteerKernel(testingObject, storageDType).Run(base, direction, coefficient, out)
	convey.So(err, convey.ShouldBeNil)
	assertUtilityStorageForTest(testingObject, backend, out, storageDType, fixture.expectedBytes)
}

func lookupActivationSteerKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("activation_steer", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			dtype.Float32,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("activation_steer Metal kernel not registered for %v", storageDType)
	}

	return kernel
}

func testNameForStorageDTypeAndElementCount(storageDType dtype.DType, elementCount int) string {
	return storageDType.String() + "/" + testNameForElementCount(elementCount)
}

func assertUtilityStorageForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		assertUtilityBytesForTest(testingObject, backend, input, storageDType, expectedBytes)

		return
	}

	assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, 1)
}
