//go:build xla

package rope_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cuprope "github.com/theapemachine/puter/device/cpu/rope"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceRoPE = cuprope.New()

func TestRoPEXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA RoPE pairs", t, func() {
		for _, count := range xlaparity.Lengths {
			if count%2 != 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("pairs=%d", count/2), func() {
				halfDim := count / 2
				input := xlaparity.RandomUnaryInput(count, 0x4100+int64(count))
				cosValues := xlaparity.RandomUnaryInput(halfDim, 0x4200+int64(count))
				sinValues := xlaparity.RandomUnaryInput(halfDim, 0x4300+int64(count))
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

				harness.Backend().RoPEPairs(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(cosTensor),
					xla.ResidentPointer(sinTensor),
					halfDim,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA RoPE", t, func() {
		seqLen := 4
		numHeads := 2
		headDim := 8
		total := seqLen * numHeads * headDim
		input := xlaparity.RandomUnaryInput(total, 0x4400)
		want := make([]float32, total)
		config := device.RoPEConfig{BaseFreq: 10000.0, StartPosition: 0}
		referenceRoPE.RoPE(config, unsafe.Pointer(&input[0]), unsafe.Pointer(&want[0]), seqLen, numHeads, headDim, dtype.Float32)

		inputTensor := harness.UploadVolume(input, seqLen, numHeads, headDim, dtype.Float32)
		outputTensor := harness.UploadVolume(make([]float32, total), seqLen, numHeads, headDim, dtype.Float32)
		defer inputTensor.Close()
		defer outputTensor.Close()

		harness.Backend().RoPE(
			config,
			xla.ResidentPointer(inputTensor),
			xla.ResidentPointer(outputTensor),
			seqLen,
			numHeads,
			headDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})
}
