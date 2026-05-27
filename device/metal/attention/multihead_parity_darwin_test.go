//go:build darwin && cgo

package attention

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpuattention "github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestMultiHeadAttentionMetalDecodeCausalMaskParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given single-token GQA decode over a longer KV cache", testingObject, func() {
		query := []float32{0, 0, 0, 0}
		key := []float32{0, 0, 0, 0, 0, 0}
		value := []float32{1, 10, 2, 20, 3, 30}
		want := []float32{2, 20, 2, 20}
		config := device.MultiHeadAttentionConfig{
			NumHeads:    2,
			KVHeadCount: 1,
			HeadDim:     2,
			Causal:      true,
		}
		got := runMetalAttention(testingObject, harness, config, query, key, value, 1, 3)

		convey.Convey("It should attend through the live cache prefix", func() {
			parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 2)
		})
	})
}

func TestMultiHeadAttentionMetalPrefillParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given nonzero GQA prefill inputs", testingObject, func() {
		config := device.MultiHeadAttentionConfig{
			NumHeads:    2,
			KVHeadCount: 1,
			HeadDim:     2,
			Causal:      true,
		}
		query := []float32{
			0.25, -0.50, 0.75, 0.10,
			-0.30, 0.20, 0.40, -0.80,
			0.60, 0.90, -0.20, 0.35,
		}
		key := []float32{
			0.10, 0.20,
			-0.40, 0.30,
			0.50, -0.70,
		}
		value := []float32{
			1, 10,
			2, 20,
			3, 30,
		}
		want := runCPUAttention(testingObject, config, query, key, value, 3, 3)
		got := runMetalAttention(testingObject, harness, config, query, key, value, 3, 3)

		convey.Convey("It should match the CPU reference", func() {
			parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 8)
		})
	})
}

func BenchmarkMultiHeadAttentionMetalDecode(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	query := make([]float32, 1*32*64)
	key := make([]float32, 128*8*64)
	value := make([]float32, 128*8*64)
	output := make([]float32, len(query))

	queryBuffer := harness.UploadVector(query, dtype.Float32)
	defer queryBuffer.Close()

	keyBuffer := harness.UploadVector(key, dtype.Float32)
	defer keyBuffer.Close()

	valueBuffer := harness.UploadVector(value, dtype.Float32)
	defer valueBuffer.Close()

	outputBuffer := harness.UploadVector(output, dtype.Float32)
	defer outputBuffer.Close()

	config := device.MultiHeadAttentionConfig{
		NumHeads:    32,
		KVHeadCount: 8,
		HeadDim:     64,
		Causal:      true,
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := DispatchMultiHeadAttentionRefs(
			harness.ContextRef(),
			queryBuffer.Ref(),
			keyBuffer.Ref(),
			valueBuffer.Ref(),
			outputBuffer.Ref(),
			config,
			1,
			128,
			dtype.Float32,
		); err != nil {
			benchmark.Fatal(err)
		}
	}

	harness.Sync()
}

func runMetalAttention(
	testingObject *testing.T,
	harness *parity.Harness,
	config device.MultiHeadAttentionConfig,
	query []float32,
	key []float32,
	value []float32,
	seqQ int,
	seqK int,
) []float32 {
	testingObject.Helper()

	queryBuffer := harness.UploadVector(query, dtype.Float32)
	defer queryBuffer.Close()

	keyBuffer := harness.UploadVector(key, dtype.Float32)
	defer keyBuffer.Close()

	valueBuffer := harness.UploadVector(value, dtype.Float32)
	defer valueBuffer.Close()

	outputBuffer := harness.UploadVector(make([]float32, len(query)), dtype.Float32)
	defer outputBuffer.Close()

	err := DispatchMultiHeadAttentionRefs(
		harness.ContextRef(),
		queryBuffer.Ref(),
		keyBuffer.Ref(),
		valueBuffer.Ref(),
		outputBuffer.Ref(),
		config,
		uint32(seqQ),
		uint32(seqK),
		dtype.Float32,
	)
	convey.So(err, convey.ShouldBeNil)

	return harness.DownloadFloat32(outputBuffer, dtype.Float32)
}

func runCPUAttention(
	testingObject *testing.T,
	config device.MultiHeadAttentionConfig,
	query []float32,
	key []float32,
	value []float32,
	seqQ int,
	seqK int,
) []float32 {
	testingObject.Helper()

	queryTensor := hostFloat32Tensor(testingObject, []int{seqQ, config.NumHeads * config.HeadDim}, query)
	keyTensor := hostFloat32Tensor(testingObject, []int{seqK, config.KVHeadCount * config.HeadDim}, key)
	valueTensor := hostFloat32Tensor(testingObject, []int{seqK, config.KVHeadCount * config.HeadDim}, value)
	outputTensor := hostFloat32Tensor(
		testingObject,
		[]int{seqQ, config.NumHeads * config.HeadDim},
		make([]float32, len(query)),
	)

	err := cpuattention.MultiHeadAttentionFloat32(config, queryTensor, keyTensor, valueTensor, outputTensor)
	convey.So(err, convey.ShouldBeNil)

	output, err := outputTensor.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	return append([]float32(nil), output...)
}

func hostFloat32Tensor(
	testingObject *testing.T,
	dimensions []int,
	values []float32,
) tensor.Tensor {
	testingObject.Helper()

	shape, err := tensor.NewShape(dimensions)
	if err != nil {
		testingObject.Fatalf("hostFloat32Tensor: shape: %v", err)
	}

	resident, err := tensor.New(shape, dtype.Float32)
	if err != nil {
		testingObject.Fatalf("hostFloat32Tensor: tensor: %v", err)
	}

	view, err := resident.Float32Native()
	if err != nil {
		testingObject.Fatalf("hostFloat32Tensor: native: %v", err)
	}

	copy(view, values)

	return resident
}
