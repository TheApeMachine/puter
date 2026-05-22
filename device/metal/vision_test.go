package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func TestKernelRegistry_MetalVisionDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalVisionDType(testingObject, backend, storageDType)
		})
	}
}

func TestMetalConv2DVAEPostQuantShape(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			convey.Convey("Given the FLUX VAE post_quant_conv tensor shapes", testingObject, func() {
				inputShape := mustShapeForTest(testingObject, []int{1, 32, 128, 128})
				weightShape := mustShapeForTest(testingObject, []int{32, 32, 1, 1})
				biasShape := mustShapeForTest(testingObject, []int{32})
				outputShape := mustShapeForTest(testingObject, []int{1, 32, 128, 128})
				inputBytes := encodeVisionValues(make([]float32, inputShape.Len()), storageDType)
				weightBytes := encodeVisionValues(make([]float32, weightShape.Len()), storageDType)
				biasBytes := encodeVisionValues(make([]float32, biasShape.Len()), storageDType)
				input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
				weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
				bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
				output := emptyTensorForTest(testingObject, backend, outputShape, storageDType)

				defer closeBenchmarkTensors(input, weight, bias, output)

				convey.Convey("It should dispatch without crashing", func() {
					err := lookupVisionConv2DKernel(testingObject, storageDType).Run(input, weight, bias, output)

					convey.So(err, convey.ShouldBeNil)
				})
			})
		})
	}
}

func TestMetalConv2DBatchedDispatch(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given conv2d dispatch inside a Metal batch", testingObject, func() {
		inputShape := mustShapeForTest(testingObject, []int{1, 32, 128, 128})
		weightShape := mustShapeForTest(testingObject, []int{32, 32, 1, 1})
		biasShape := mustShapeForTest(testingObject, []int{32})
		outputShape := mustShapeForTest(testingObject, []int{1, 32, 128, 128})
		input := uploadDTypeTensorForTest(
			testingObject,
			backend,
			inputShape,
			dtype.Float32,
			encodeVisionValues(make([]float32, inputShape.Len()), dtype.Float32),
		)
		weight := uploadDTypeTensorForTest(
			testingObject,
			backend,
			weightShape,
			dtype.Float32,
			encodeVisionValues(make([]float32, weightShape.Len()), dtype.Float32),
		)
		bias := uploadDTypeTensorForTest(
			testingObject,
			backend,
			biasShape,
			dtype.Float32,
			encodeVisionValues(make([]float32, biasShape.Len()), dtype.Float32),
		)
		firstOutput := emptyTensorForTest(testingObject, backend, outputShape, dtype.Float32)
		secondOutput := emptyTensorForTest(testingObject, backend, outputShape, dtype.Float32)

		defer closeBenchmarkTensors(input, weight, bias, firstOutput, secondOutput)

		convey.Convey("It should keep the batch encoder valid across vision kernels", func() {
			backend.BeginBatch()
			defer func() {
				if err := backend.EndBatch(); err != nil {
					testingObject.Fatalf("EndBatch failed: %v", err)
				}
			}()

			kernel := lookupVisionConv2DKernel(testingObject, dtype.Float32)
			firstErr := kernel.Run(input, weight, bias, firstOutput)
			secondErr := kernel.Run(input, weight, bias, secondOutput)

			convey.So(firstErr, convey.ShouldBeNil)
			convey.So(secondErr, convey.ShouldBeNil)
		})
	})
}

func TestMetalConv2DSamePadding(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given a conv2d output shape that implies same padding", testingObject, func() {
		inputShape := mustShapeForTest(testingObject, []int{1, 1, 3, 3})
		weightShape := mustShapeForTest(testingObject, []int{1, 1, 3, 3})
		biasShape := mustShapeForTest(testingObject, []int{1})
		outputShape := mustShapeForTest(testingObject, []int{1, 1, 3, 3})
		input := uploadDTypeTensorForTest(
			testingObject,
			backend,
			inputShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}),
		)
		weight := uploadDTypeTensorForTest(
			testingObject,
			backend,
			weightShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1}),
		)
		bias := uploadDTypeTensorForTest(
			testingObject,
			backend,
			biasShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{0}),
		)
		output := emptyTensorForTest(testingObject, backend, outputShape, dtype.Float32)

		defer closeBenchmarkTensors(input, weight, bias, output)

		convey.Convey("It should use zero padding around the input", func() {
			err := lookupVisionConv2DKernel(testingObject, dtype.Float32).Run(input, weight, bias, output)

			convey.So(err, convey.ShouldBeNil)
			assertProjectionBytesForTest(
				testingObject,
				backend,
				output,
				dtype.Float32,
				convert.Float32ToBytes([]float32{12, 21, 16, 27, 45, 33, 24, 39, 28}),
			)
		})
	})
}

func encodeVisionValues(values []float32, storageDType dtype.DType) []byte {
	switch storageDType {
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for valueIndex, value := range values {
			encoded[valueIndex] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded)
	case dtype.BFloat16:
		var bf16 dtype.BF16

		return bf16.EncodeFloat32(values)
	default:
		return convert.Float32ToBytes(values)
	}
}

func testMetalVisionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, outputWidth := range parityElementCounts {
		outputWidth := outputWidth

		testingObject.Run(testNameForElementCount(outputWidth), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" vision tensors", testingObject, func() {
				runConv2DParityCase(testingObject, backend, storageDType, outputWidth)
			})
		})
	}
}

func runConv2DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outputWidth int,
) {
	inputBytes, weightBytes, biasBytes, expectedBytes := conv2DDTypeBytes(outputWidth, storageDType)
	input, weight, bias, out := conv2DTensorsForTest(
		testingObject, backend, outputWidth, storageDType, inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupVisionConv2DKernel(testingObject, storageDType).Run(input, weight, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func lookupVisionConv2DKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("conv2d", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s conv2d kernel", storageDType.Name())
	}

	return kernel
}

func lookupVisionPool2DKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func conv2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 4, 2, 3
	outputHeight := inputHeight - kernelHeight + 1
	inputWidth := outputWidth + kernelWidth - 1
	inputShape := mustShapeForTest(testingObject, []int{batch, inChannels, inputHeight, inputWidth})
	weightShape := mustShapeForTest(testingObject, []int{outChannels, inChannels, kernelHeight, kernelWidth})
	biasShape := mustShapeForTest(testingObject, []int{outChannels})
	outShape := mustShapeForTest(testingObject, []int{batch, outChannels, outputHeight, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, weight, bias, out
}

func pool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels, inputHeight := 2, 3, 4
	inputWidth := outputWidth * 2
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight / 2, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	maxOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	avgOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, maxOut, avgOut
}

func adaptivePool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels, inputHeight := 2, 2, 5
	inputWidth := outputWidth + 3
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, 3, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	avgOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	maxOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, avgOut, maxOut
}
