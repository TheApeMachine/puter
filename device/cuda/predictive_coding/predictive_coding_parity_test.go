//go:build cuda

package predictive_coding

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpupc "github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestPredictiveCodingCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA prediction", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				outDim := count
				inDim := count
				weights := parity.RandomUnaryInput(outDim*inDim, 0xB000+int64(count))
				state := parity.RandomUnaryInput(inDim, 0xB001+int64(count))
				want := make([]float32, outDim)
				cpupc.PredictionFloat32Scalar(weights, state, want, outDim, inDim)

				weightsTensor := harness.UploadVector(weights, dtype.Float32)
				stateTensor := harness.UploadVector(state, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, outDim), dtype.Float32)
				defer weightsTensor.Close()
				defer stateTensor.Close()
				defer outputTensor.Close()

				if err := DispatchPrediction(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(weightsTensor.Ref()),
					cudadevice.BufferRef(stateTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(outDim),
					uint32(inDim),
				); err != nil {
					t.Fatalf("dispatch Prediction: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
