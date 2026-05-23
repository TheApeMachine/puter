//go:build cuda

package rope

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cuprope "github.com/theapemachine/puter/device/cpu/rope"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestRoPECUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA RoPE pairs", t, func() {
		for _, count := range parity.Lengths {
			if count%2 != 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("pairs=%d", count/2), func() {
				halfDim := uint32(count / 2)
				input := parity.RandomUnaryInput(count, 0xR000+int64(count))
				cosValues := parity.RandomUnaryInput(int(halfDim), 0xR100+int64(count))
				sinValues := parity.RandomUnaryInput(int(halfDim), 0xR200+int64(count))
				want := make([]float32, count)
				cuprope.RopePairsGeneric(want, input, cosValues, sinValues)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				cosTensor := harness.UploadVector(cosValues, dtype.Float32)
				sinTensor := harness.UploadVector(sinValues, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer cosTensor.Close()
				defer sinTensor.Close()
				defer outputTensor.Close()

				if err := DispatchRoPEPairs(
					parity.DeviceRef(harness.ContextRef()),
					parity.BufferRef(inputTensor.Ref()),
					parity.BufferRef(outputTensor.Ref()),
					parity.BufferRef(cosTensor.Ref()),
					parity.BufferRef(sinTensor.Ref()),
					halfDim,
					dtype.Float32,
				); err != nil {
					t.Fatalf("dispatch RoPEPairs: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
