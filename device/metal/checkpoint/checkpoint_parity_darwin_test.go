//go:build darwin && cgo

package checkpoint

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpucheckpoint "github.com/theapemachine/puter/device/cpu/checkpoint"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestCheckpointEncodeDecodeMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal checkpoint encode/decode kernels", testingObject, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				inputValues := parity.RandomUnaryInput(count, 0x5400+int64(count))
				dims := []int{count}
				headerBytes := CheckpointHeaderBytes(len(dims))
				dataBytes := count * 4
				wantPayload := checkpointPayloadReference(inputValues)

				inputTensor := harness.UploadVector(inputValues, dtype.Float32)
				outputTensor := harness.UploadBytes(make([]byte, int(headerBytes)+dataBytes))
				defer inputTensor.Close()
				defer outputTensor.Close()

				encodeErr := DispatchCheckpointEncodeRefs(
					harness.ContextRef(),
					inputTensor.Ref(),
					outputTensor.Ref(),
					uint32(len(dims)),
					uint32(count),
					dimsToUint64(dims),
				)
				convey.So(encodeErr, convey.ShouldBeNil)

				harness.Sync()
				encodedBytes := outputTensor.ReadBytes()
				convey.So(encodedBytes[headerBytes:], convey.ShouldResemble, wantPayload)

				decodeInput := harness.UploadBytes(encodedBytes)
				decodeOutput := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer decodeInput.Close()
				defer decodeOutput.Close()

				decodeErr := DispatchCheckpointDecodeRefs(
					harness.ContextRef(),
					decodeInput.Ref(),
					decodeOutput.Ref(),
					uint32(headerBytes),
					uint32(count),
				)
				convey.So(decodeErr, convey.ShouldBeNil)

				got := harness.DownloadFloat32(decodeOutput, dtype.Float32)
				cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, inputValues, 0)
			})
		}
	})
}

func BenchmarkCheckpointEncodeMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	count := 8192
	input := parity.RandomUnaryInput(count, 0x5420)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadBytes(make([]byte, int(CheckpointHeaderBytes(1))+count*4))
	defer inputTensor.Close()
	defer outputTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchCheckpointEncodeRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			outputTensor.Ref(),
			1,
			uint32(count),
			[]uint64{uint64(count)},
		)
	}

	harness.Sync()
}

func checkpointPayloadReference(values []float32) []byte {
	payload := make([]byte, len(values)*4)
	cpucheckpoint.EncodeFloat32DataNative(payload, values)

	return payload
}

func dimsToUint64(dims []int) []uint64 {
	converted := make([]uint64, len(dims))

	for index, dimension := range dims {
		converted[index] = uint64(dimension)
	}

	return converted
}
