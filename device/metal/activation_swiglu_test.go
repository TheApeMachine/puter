//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

const swiGLUMaxULP uint32 = 1

func TestKernelRegistry_MetalSwiGLUDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalSwiGLUDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" tensors for swiglu",
						testingObject,
						func() {
							runSwiGLUParityCase(testingObject, backend, storageDType, elementCount)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalPackedSwiGLUDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalSwiGLUDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			convey.Convey(
				"Given Metal "+storageDType.Name()+" tensors for packed swiglu",
				testingObject,
				func() {
					runPackedSwiGLUParityCase(testingObject, backend, storageDType, 2, 7)
				},
			)
		})
	}
}

func runSwiGLUParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := swiGLUFixtureForTest(testingObject, backend, elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	gate := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.gateBytes)
	up := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.upBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(gate, up, out)

	err := lookupSwiGLUKernel(testingObject, storageDType).Run(gate, up, out)
	convey.So(err, convey.ShouldBeNil)

	if storageDType == dtype.Float32 {
		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(fixture.expectedBytes),
			swiGLUMaxULP,
		)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, swiGLUMaxULP)
}

func runPackedSwiGLUParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	rows int,
	inner int,
) {
	fixture := swiGLUFixtureForTest(testingObject, backend, rows*inner, storageDType)
	packedShape := mustShapeForTest(testingObject, []int{rows, inner * 2})
	outputShape := mustShapeForTest(testingObject, []int{rows, inner})
	packedBytes := packedSwiGLUBytesForTest(testingObject, storageDType, fixture, rows, inner)
	packed := uploadDTypeTensorForTest(testingObject, backend, packedShape, storageDType, packedBytes)
	out := emptyTensorForTest(testingObject, backend, outputShape, storageDType)
	defer closeBenchmarkTensors(packed, out)

	err := lookupPackedSwiGLUKernel(testingObject, storageDType).Run(packed, out)
	convey.So(err, convey.ShouldBeNil)

	if storageDType == dtype.Float32 {
		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(fixture.expectedBytes),
			swiGLUMaxULP,
		)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, swiGLUMaxULP)
}

func packedSwiGLUBytesForTest(
	testingObject testing.TB,
	storageDType dtype.DType,
	fixture swiGLUFixture,
	rows int,
	inner int,
) []byte {
	testingObject.Helper()

	elementBytes, err := storageDType.Size()
	if err != nil {
		testingObject.Fatalf("dtype size failed: %v", err)
	}

	packed := make([]byte, len(fixture.gateBytes)+len(fixture.upBytes))

	for row := range rows {
		for column := range inner {
			source := (row*inner + column) * elementBytes
			target := (row*inner*2 + column) * elementBytes
			copy(packed[target:target+elementBytes], fixture.gateBytes[source:source+elementBytes])

			target = (row*inner*2 + inner + column) * elementBytes
			copy(packed[target:target+elementBytes], fixture.upBytes[source:source+elementBytes])
		}
	}

	return packed
}

func lookupSwiGLUKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("swiglu", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("swiglu Metal kernel not registered for %s", storageDType.Name())
	}

	return kernel
}

func lookupPackedSwiGLUKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("swiglu", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("packed swiglu Metal kernel not registered for %s", storageDType.Name())
	}

	return kernel
}
