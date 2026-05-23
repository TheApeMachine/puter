//go:build cuda

package layernorm

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func randomLayerNormVector(count int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = rng.Float32()*2.0 - 1.0
	}

	return values
}

func TestLayerNormCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA LayerNorm", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("cols=%d", count), func() {
				rows := 1
				elementCount := rows * count
				input := randomLayerNormVector(elementCount, 0x4F00+int64(count))
				scale := randomLayerNormVector(count, 0x4F01+int64(count))
				bias := randomLayerNormVector(count, 0x4F02+int64(count))
				want := parity.LayerNormReference(input, scale, bias, rows, count, dtype.Float32)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				scaleTensor := harness.UploadVector(scale, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
				defer inputTensor.Close()
				defer scaleTensor.Close()
				defer biasTensor.Close()
				defer outputTensor.Close()

				if err := DispatchLayerNorm(
					parity.DeviceRef(harness.ContextRef()),
					parity.BufferRef(inputTensor.Ref()),
					parity.BufferRef(scaleTensor.Ref()),
					parity.BufferRef(biasTensor.Ref()),
					parity.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(rows),
					uint32(count),
					0,
				); err != nil {
					t.Fatalf("dispatch LayerNorm: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 3)
			})
		}
	})
}
