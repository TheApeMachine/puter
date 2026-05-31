//go:build darwin && cgo

package optimizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

const optimizerMetalMaxULP = 2

func TestAdamMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal Adam kernels", testingObject, func() {
		config := cpuoptimizer.DefaultAdamConfig()

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				params := parity.RandomUnaryInput(count, 0x5500+int64(count))
				gradients := parity.RandomUnaryInput(count, 0x5501+int64(count))
				firstMoment := parity.RandomUnaryInput(count, 0x5502+int64(count))
				secondMoment := parity.RandomUnaryInput(count, 0x5503+int64(count))

				wantParams, _, _ := adamCPUReference(
					config,
					params,
					gradients,
					firstMoment,
					secondMoment,
					dtype.Float32,
				)

				paramsTensor := harness.UploadVector(params, dtype.Float32)
				gradientsTensor := harness.UploadVector(gradients, dtype.Float32)
				firstTensor := harness.UploadVector(firstMoment, dtype.Float32)
				secondTensor := harness.UploadVector(secondMoment, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer paramsTensor.Close()
				defer gradientsTensor.Close()
				defer firstTensor.Close()
				defer secondTensor.Close()
				defer outputTensor.Close()

				dispatchErr := DispatchOptimizer4Refs(
					harness.ContextRef(),
					OperationAdam,
					paramsTensor.Ref(),
					gradientsTensor.Ref(),
					firstTensor.Ref(),
					secondTensor.Ref(),
					outputTensor.Ref(),
					dtype.Float32,
					uint32(count),
					AdamMetalConfig(config),
				)
				convey.So(dispatchErr, convey.ShouldBeNil)

				gotParams := harness.DownloadFloat32(outputTensor, dtype.Float32)

				cpuparity.AssertFloat32SlicesWithinULP(
					testingObject,
					gotParams,
					wantParams,
					optimizerMetalMaxULP,
				)
			})
		}
	})
}

func BenchmarkAdamMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	config := cpuoptimizer.DefaultAdamConfig()
	count := 8192
	params := parity.RandomUnaryInput(count, 0x5510)
	gradients := parity.RandomUnaryInput(count, 0x5511)
	firstMoment := parity.RandomUnaryInput(count, 0x5512)
	secondMoment := parity.RandomUnaryInput(count, 0x5513)

	paramsTensor := harness.UploadVector(params, dtype.Float32)
	gradientsTensor := harness.UploadVector(gradients, dtype.Float32)
	firstTensor := harness.UploadVector(firstMoment, dtype.Float32)
	secondTensor := harness.UploadVector(secondMoment, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer paramsTensor.Close()
	defer gradientsTensor.Close()
	defer firstTensor.Close()
	defer secondTensor.Close()
	defer outputTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchOptimizer4Refs(
			harness.ContextRef(),
			OperationAdam,
			paramsTensor.Ref(),
			gradientsTensor.Ref(),
			firstTensor.Ref(),
			secondTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(count),
			AdamMetalConfig(config),
		)
	}

	harness.Sync()
}

func adamCPUReference(
	config cpuoptimizer.AdamConfig,
	params, gradients, firstMoment, secondMoment []float32,
	format dtype.DType,
) ([]float32, []float32, []float32) {
	hostBackend := tensor.NewHostBackend()
	shape, shapeErr := tensor.NewShape([]int{len(params)})

	if shapeErr != nil {
		panic(shapeErr)
	}

	paramsTensor, uploadErr := hostBackend.Upload(shape, format, encodeNative(params, format))

	if uploadErr != nil {
		panic(uploadErr)
	}

	gradientsTensor, uploadErr := hostBackend.Upload(shape, format, encodeNative(gradients, format))

	if uploadErr != nil {
		panic(uploadErr)
	}

	firstTensor, uploadErr := hostBackend.Upload(shape, dtype.Float32, convert.Float32ToBytes(firstMoment))

	if uploadErr != nil {
		panic(uploadErr)
	}

	secondTensor, uploadErr := hostBackend.Upload(shape, dtype.Float32, convert.Float32ToBytes(secondMoment))

	if uploadErr != nil {
		panic(uploadErr)
	}

	outputTensor, uploadErr := hostBackend.Upload(shape, format, encodeNative(make([]float32, len(params)), format))

	if uploadErr != nil {
		panic(uploadErr)
	}

	stepErr := cpuoptimizer.AdamStepFloat32Scalar(
		config,
		paramsTensor,
		gradientsTensor,
		firstTensor,
		secondTensor,
		outputTensor,
	)

	if stepErr != nil {
		panic(stepErr)
	}

	wantParams := decodeNative(outputTensor, format)
	wantFirst, nativeErr := firstTensor.Float32Native()

	if nativeErr != nil {
		panic(nativeErr)
	}

	wantSecond, nativeErr := secondTensor.Float32Native()

	if nativeErr != nil {
		panic(nativeErr)
	}

	return wantParams, wantFirst, wantSecond
}

func encodeNative(values []float32, format dtype.DType) []byte {
	switch format {
	case dtype.Float32:
		return convert.Float32ToBytes(values)
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for index, value := range values {
			encoded[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded)
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded)
	default:
		panic(fmt.Sprintf("unsupported dtype %v", format))
	}
}

func decodeNative(resident tensor.Tensor, format dtype.DType) []float32 {
	switch format {
	case dtype.Float32:
		values, err := resident.Float32Native()

		if err != nil {
			panic(err)
		}

		return values
	case dtype.Float16:
		values, err := resident.Float16Native()

		if err != nil {
			panic(err)
		}

		decoded := make([]float32, len(values))

		for index, value := range values {
			decoded[index] = value.Float32()
		}

		return decoded
	case dtype.BFloat16:
		values, err := resident.BFloat16Native()

		if err != nil {
			panic(err)
		}

		decoded := make([]float32, len(values))

		for index, value := range values {
			decoded[index] = value.Float32()
		}

		return decoded
	default:
		panic(fmt.Sprintf("unsupported dtype %v", format))
	}
}
