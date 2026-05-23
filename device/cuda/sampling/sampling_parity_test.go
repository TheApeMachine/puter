//go:build cuda

package sampling

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func greedySampleReference(logits []float32) int32 {
	bestIndex := int32(0)
	bestValue := logits[0]

	for index := 1; index < len(logits); index++ {
		if logits[index] > bestValue {
			bestValue = logits[index]
			bestIndex = int32(index)
		}
	}

	return bestIndex
}

func TestSamplingCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA greedy sampling", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				logits := parity.RandomUnaryInput(count, 0x5A00+int64(count))
				want := []float32{float32(greedySampleReference(logits))}

				logitsTensor := harness.UploadVector(logits, dtype.Float32)
				scoresTensor := harness.UploadVector(make([]float32, PaddedCount(uint32(count))), dtype.Float32)
				indicesTensor := harness.UploadVector(make([]float32, PaddedCount(uint32(count))), dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer logitsTensor.Close()
				defer scoresTensor.Close()
				defer indicesTensor.Close()
				defer outputTensor.Close()

				if err := DispatchSampling(
					cudadevice.DeviceRef(harness.ContextRef()),
					0,
					cudadevice.BufferRef(logitsTensor.Ref()),
					cudadevice.BufferRef(scoresTensor.Ref()),
					cudadevice.BufferRef(indicesTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
					0,
				); err != nil {
					t.Fatalf("dispatch Greedy: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
