//go:build darwin && cgo

package metal

import (
	"context"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	cpurope "github.com/theapemachine/puter/device/cpu/rope"
)

func TestRoPEMetalLlama3HalfModeParity(testingObject *testing.T) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	convey.Convey("Given Llama 3 half-mode RoPE inputs", testingObject, func() {
		config := llama3HalfModeRoPEConfig()
		seqLen := 4
		numHeads := 2
		headDim := 8
		input := randomRoPEInput(seqLen*numHeads*headDim, 0x9120)
		want := make([]float32, len(input))

		cpurope.Default.RoPE(
			config,
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&want[0]),
			seqLen,
			numHeads,
			headDim,
			dtype.Float32,
		)

		got := runMetalRoPE(testingObject, backend, config, input, seqLen, numHeads, headDim)

		convey.Convey("It should match the CPU reference", func() {
			cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 8)
		})
	})
}

func BenchmarkRoPEMetalLlama3HalfMode(benchmark *testing.B) {
	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}
	defer backend.Close()

	config := llama3HalfModeRoPEConfig()
	seqLen := 128
	numHeads := 32
	headDim := 64
	input := make([]float32, seqLen*numHeads*headDim)
	output := make([]float32, len(input))
	inputTensor := uploadRoPETensor(benchmark, backend, input)
	defer inputTensor.Close()
	outputTensor := uploadRoPETensor(benchmark, backend, output)
	defer outputTensor.Close()
	inputPointer := inputTensor.DispatchPointer()
	outputPointer := outputTensor.DispatchPointer()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.RoPE(
			config,
			inputPointer,
			outputPointer,
			seqLen,
			numHeads,
			headDim,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func runMetalRoPE(
	testingObject *testing.T,
	backend *Backend,
	config device.RoPEConfig,
	input []float32,
	seqLen int,
	numHeads int,
	headDim int,
) []float32 {
	testingObject.Helper()

	inputTensor := uploadRoPETensor(testingObject, backend, input)
	defer inputTensor.Close()
	outputTensor := uploadRoPETensor(testingObject, backend, make([]float32, len(input)))
	defer outputTensor.Close()

	backend.RoPE(
		config,
		inputTensor.DispatchPointer(),
		outputTensor.DispatchPointer(),
		seqLen,
		numHeads,
		headDim,
		dtype.Float32,
	)
	backend.SyncDevice()

	dataType, rawBytes, err := outputTensor.RawBytes()
	convey.So(err, convey.ShouldBeNil)

	got, err := convert.BytesToFloat32(dataType, rawBytes)
	convey.So(err, convey.ShouldBeNil)

	return got
}

func llama3HalfModeRoPEConfig() device.RoPEConfig {
	return device.RoPEConfig{
		BaseFreq:        500000.0,
		StartPosition:   3,
		Mode:            device.RoPEModeHalf,
		Scaling:         device.RoPEScalingLlama3,
		ScalingFactor:   32.0,
		LowFreqFactor:   1.0,
		HighFreqFactor:  4.0,
		OriginalContext: 8192,
	}
}

func uploadRoPETensor(
	testingHandle interface {
		Helper()
		Fatalf(string, ...any)
	},
	backend *Backend,
	values []float32,
) *DeviceTensor {
	testingHandle.Helper()

	shape, err := tensor.NewShape([]int{len(values)})
	if err != nil {
		testingHandle.Fatalf("uploadRoPETensor: shape: %v", err)
	}

	resident, err := backend.Upload(shape, dtype.Float32, convert.Float32ToBytes(values))
	if err != nil {
		testingHandle.Fatalf("uploadRoPETensor: upload: %v", err)
	}

	deviceTensor, ok := resident.(*DeviceTensor)
	if !ok {
		testingHandle.Fatalf("uploadRoPETensor: got %T", resident)
	}

	return deviceTensor
}

func randomRoPEInput(count int, seed int64) []float32 {
	generator := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = generator.Float32()*4.0 - 2.0
	}

	return values
}
