//go:build cuda

package layernorm

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkLayerNormCUDAF32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	rows := 32
	cols := 8192
	elementCount := rows * cols
	input := randomLayerNormVector(elementCount, 1)
	scale := randomLayerNormVector(cols, 2)
	bias := randomLayerNormVector(cols, 3)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	scaleTensor := harness.UploadVector(scale, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchLayerNormRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			scaleTensor.Ref(),
			biasTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(rows),
			uint32(cols),
			0,
		); err != nil {
			b.Fatal(err)
		}
	}
}
