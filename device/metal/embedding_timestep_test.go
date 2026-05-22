//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

const timestepEmbeddingMaxULP uint32 = 128

func TestKernelRegistry_MetalTimestepEmbedding(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalProjectionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for timestep embedding", testingObject, func() {
				runTimestepEmbeddingParityCase(testingObject, backend, storageDType)
			})
		})
	}
}

func runTimestepEmbeddingParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) {
	timestepShape := mustShapeForTest(testingObject, []int{2})
	outputShape := mustShapeForTest(testingObject, []int{2, 8})
	timesteps := []float32{0.5, 999.0}
	input := uploadDTypeTensorForTest(
		testingObject,
		backend,
		timestepShape,
		dtype.Float32,
		dtypeconvert.Float32ToBytes(timesteps),
	)
	maxPeriod := uploadFloat32ScalarForTest(testingObject, backend, 10000)
	downscale := uploadFloat32ScalarForTest(testingObject, backend, 0)
	flip := uploadInt32ScalarForTest(testingObject, backend, 1)
	out := emptyTensorForTest(testingObject, backend, outputShape, storageDType)
	defer closeBenchmarkTensors(input, maxPeriod, downscale, flip, out)

	err := lookupTimestepKernel(testingObject, storageDType).Run(input, maxPeriod, downscale, flip, out)
	convey.So(err, convey.ShouldBeNil)

	expectedBytes := timestepEmbeddingExpectedBytes(timesteps, 8, 10000, 0, true, storageDType)

	if storageDType == dtype.Float32 {
		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(expectedBytes),
			timestepEmbeddingMaxULP,
		)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, timestepEmbeddingMaxULP)
}

func lookupTimestepKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("timestep", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			dtype.Float32,
			dtype.Float32,
			dtype.Float32,
			dtype.Int32,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s timestep kernel", storageDType.Name())
	}

	return kernel
}

func timestepEmbeddingExpectedBytes(
	timesteps []float32,
	dim int,
	maxPeriod float32,
	downscaleFreqShift float32,
	flipSinToCos bool,
	storageDType dtype.DType,
) []byte {
	half := dim / 2
	values := make([]float32, len(timesteps)*dim)

	for row, timestep := range timesteps {
		for column := range half {
			exponent := -math.Log(float64(maxPeriod)) * float64(column)
			exponent /= float64(half) - float64(downscaleFreqShift)
			angle := float64(timestep) * math.Exp(exponent)
			sinValue := float32(math.Sin(angle))
			cosValue := float32(math.Cos(angle))

			if flipSinToCos {
				values[row*dim+column] = cosValue
				values[row*dim+half+column] = sinValue
				continue
			}

			values[row*dim+column] = sinValue
			values[row*dim+half+column] = cosValue
		}
	}

	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func uploadFloat32ScalarForTest(
	testingObject testing.TB,
	backend *Backend,
	value float32,
) tensor.Tensor {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{1})
	return uploadDTypeTensorForTest(
		testingObject,
		backend,
		shape,
		dtype.Float32,
		dtypeconvert.Float32ToBytes([]float32{value}),
	)
}
