//go:build cuda

package quant

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuquant "github.com/theapemachine/puter/device/cpu/quant"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestQuantCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA int8 quant", t, func() {
		scale := float32(0.0875)
		zeroPoint := int8(-13)

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := parity.RandomUnaryInput(count, 0x5100+int64(count))
				want := make([]int8, count)
				cpuquant.QuantInt8Native(want, source, scale, zeroPoint)

				sourceTensor := harness.UploadVector(source, dtype.Float32)
				destinationTensor := harness.UploadBytes(make([]byte, count))
				defer sourceTensor.Close()
				defer destinationTensor.Close()

				if err := DispatchQuantRefs(
					harness.ContextRef(),
					sourceTensor.Ref(),
					destinationTensor.Ref(),
					scale,
					zeroPoint,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Quant: %v", err)
				}

				harness.Sync()
				got := bytesToInt8(destinationTensor.ReadBytes())

				if len(got) != len(want) {
					t.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
				}

				for index := range got {
					if got[index] != want[index] {
						t.Fatalf("lane %d got=%d want=%d", index, got[index], want[index])
					}
				}
			})
		}
	})
}

func bytesToInt8(bytesIn []byte) []int8 {
	values := make([]int8, len(bytesIn))

	for index, value := range bytesIn {
		values[index] = int8(value)
	}

	return values
}
