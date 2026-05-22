package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func TestKernelRegistry_MetalNorm3DDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalNormalizationDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalNorm3DDType(testingObject, backend, storageDType)
		})
	}
}

func TestMetalGroupNormNCHW(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given Metal Float32 NCHW groupnorm tensors", testingObject, func() {
		batch, channels, height, width := 2, 64, 4, 16
		spatial := height * width
		fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
		flatShape := mustShapeForTest(testingObject, []int{batch, channels, spatial})
		inputShape := mustShapeForTest(testingObject, []int{batch, channels, height, width})
		paramShape := mustShapeForTest(testingObject, []int{channels})
		flatInput := uploadDTypeTensorForTest(
			testingObject, backend, flatShape, dtype.Float32, fixture.inputBytes,
		)
		input := uploadDTypeTensorForTest(
			testingObject, backend, inputShape, dtype.Float32, fixture.inputBytes,
		)
		scale := uploadDTypeTensorForTest(
			testingObject, backend, paramShape, dtype.Float32, fixture.scaleBytes,
		)
		bias := uploadDTypeTensorForTest(
			testingObject, backend, paramShape, dtype.Float32, fixture.biasBytes,
		)
		flatOutput := emptyTensorForTest(testingObject, backend, flatShape, dtype.Float32)
		out := emptyTensorForTest(testingObject, backend, inputShape, dtype.Float32)

		defer closeBenchmarkTensors(flatInput, input, scale, bias, flatOutput, out)

		convey.Convey("It should flatten trailing spatial dimensions", func() {
			kernel := lookupNorm3DKernel(testingObject, "groupnorm", dtype.Float32)
			flatErr := kernel.Run(flatInput, scale, bias, flatOutput)
			err := kernel.Run(input, scale, bias, out)

			convey.So(flatErr, convey.ShouldBeNil)
			convey.So(err, convey.ShouldBeNil)
			_, flatBytes, flatDownloadErr := backend.Download(flatOutput)
			_, bytes, downloadErr := backend.Download(out)

			convey.So(flatDownloadErr, convey.ShouldBeNil)
			convey.So(downloadErr, convey.ShouldBeNil)
			convey.So(bytes, convey.ShouldResemble, flatBytes)
		})
	})
}

func testMetalNorm3DDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, spatial := range parityElementCounts {
		spatial := spatial

		testingObject.Run(testNameForElementCount(spatial), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" NCS normalization tensors", testingObject, func() {
				runGroupNormParityCase(testingObject, backend, storageDType, spatial)
				runInstanceNormParityCase(testingObject, backend, storageDType, spatial)
				runBatchNormEvalParityCase(testingObject, backend, storageDType, spatial)
			})
		})
	}
}

func runGroupNormParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	spatial int,
) {
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, storageDType)
	input, scale, bias, out := norm3DAffineTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupNorm3DKernel(testingObject, "groupnorm", storageDType).Run(input, scale, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormalizationNorm3DBytesForTest(testingObject, backend, out, storageDType, fixture.groupBytes)
}

func runInstanceNormParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	spatial int,
) {
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, storageDType)
	input, scale, bias, out := norm3DAffineTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupNorm3DKernel(testingObject, "instancenorm", storageDType).Run(input, scale, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormalizationNorm3DBytesForTest(testingObject, backend, out, storageDType, fixture.instanceBytes)
}

func runBatchNormEvalParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	spatial int,
) {
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, storageDType)
	input, scale, bias, mean, variance, out := batchNormEvalTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, mean, variance, out)

	err := lookupBatchNormEvalKernel(testingObject, storageDType).Run(input, scale, bias, mean, variance, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormalizationNorm3DBytesForTest(testingObject, backend, out, storageDType, fixture.batchBytes)
}

func lookupNorm3DKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func lookupBatchNormEvalKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("batchnorm_eval", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s batchnorm_eval kernel", storageDType.Name())
	}

	return kernel
}

func norm3DAffineTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	batch int,
	channels int,
	spatial int,
	fixture norm3DFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, spatial})
	paramShape := mustShapeForTest(testingObject, []int{channels})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	scale := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, fixture.scaleBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, fixture.biasBytes)
	out := emptyTensorForTest(testingObject, backend, inputShape, storageDType)
	return input, scale, bias, out
}

func batchNormEvalTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	batch int,
	channels int,
	spatial int,
	fixture norm3DFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	input, scale, bias, out := norm3DAffineTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	paramShape := mustShapeForTest(testingObject, []int{channels})
	mean := uploadDTypeTensorForTest(testingObject, backend, paramShape, storageDType, fixture.meanBytes)
	variance := uploadDTypeTensorForTest(
		testingObject, backend, paramShape, storageDType, fixture.varianceBytes,
	)
	return input, scale, bias, mean, variance, out
}

func norm3DShape() (int, int) {
	return 2, 64
}
