//go:build darwin && cgo

package layernorm

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormMetalParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given Metal LayerNorm kernels", t, func() {
		configs := []struct {
			rows int
			cols int
		}{
			{rows: 4, cols: 64},
			{rows: 1, cols: 1024},
			{rows: 7, cols: 8192},
		}

		for _, config := range configs {
			config := config

			convey.Convey(fmt.Sprintf("rows=%d cols=%d", config.rows, config.cols), func() {
				for _, storageDType := range []dtype.DType{
					dtype.Float32,
					dtype.Float16,
					dtype.BFloat16,
				} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						elementCount := config.rows * config.cols
						seedBase := int64(0x4F00 + config.rows*1000 + config.cols)

						input := randomLayerNormVector(elementCount, seedBase)
						scale := randomLayerNormVector(config.cols, seedBase+1)
						bias := randomLayerNormVector(config.cols, seedBase+2)
						want := parity.LayerNormReference(
							input,
							scale,
							bias,
							config.rows,
							config.cols,
							storageDType,
						)

						inputTensor := harness.UploadVector(input, storageDType)
						scaleTensor := harness.UploadVector(scale, storageDType)
						biasTensor := harness.UploadVector(bias, storageDType)
						outputTensor := harness.UploadVector(make([]float32, elementCount), storageDType)
						defer inputTensor.Close()
						defer scaleTensor.Close()
						defer biasTensor.Close()
						defer outputTensor.Close()

						if err := DispatchLayerNormRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							scaleTensor.Ref(),
							biasTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(config.rows),
							uint32(config.cols),
						); err != nil {
							t.Fatalf("dispatch LayerNorm: %v", err)
						}

						got := harness.DownloadFloat32(outputTensor, storageDType)
						maxULP := layerNormMaxULP(storageDType)
						parity.AssertFloat32SlicesWithinULP(t, got, want, maxULP)
					})
				}
			})
		}
	})
}

func BenchmarkLayerNormMetalFloat32(b *testing.B) {
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
		); err != nil {
			b.Fatal(err)
		}
	}
}

func randomLayerNormVector(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = rng.Float32()*4.0 - 2.0
	}

	return values
}

func layerNormMaxULP(format dtype.DType) int {
	if format == dtype.Float32 {
		return 2
	}

	return 24
}
