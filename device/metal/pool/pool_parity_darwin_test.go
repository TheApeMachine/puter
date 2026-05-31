//go:build darwin && cgo

package pool

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	cpupool "github.com/theapemachine/puter/device/cpu/pool"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestMaxPool2DMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	config := cpupool.DefaultPoolConfig()

	convey.Convey("Given Metal MaxPool2D kernels", testingObject, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				inHeight, inWidth, outHeight, outWidth := poolDims(count, config)
				batch := 1
				channels := 1
				input := parity.RandomUnaryInput(
					batch*channels*inHeight*inWidth,
					0x5600+int64(count),
				)
				want := make([]float32, batch*channels*outHeight*outWidth)

				cpupool.Default.MaxPool2D(
					config,
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					batch,
					channels,
					inHeight,
					inWidth,
					outHeight,
					outWidth,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, len(want)), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				dispatchErr := DispatchPool2DRefs(
					harness.ContextRef(),
					inputTensor.Ref(),
					outputTensor.Ref(),
					dtype.Float32,
					uint32(batch),
					uint32(channels),
					uint32(inHeight),
					uint32(inWidth),
					uint32(outHeight),
					uint32(outWidth),
					true,
					false,
				)
				convey.So(dispatchErr, convey.ShouldBeNil)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 0)
			})
		}
	})
}

func BenchmarkMaxPool2DMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	config := cpupool.DefaultPoolConfig()
	inHeight, inWidth, outHeight, outWidth := poolDims(8192, config)
	input := parity.RandomUnaryInput(inHeight*inWidth, 0x5610)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, outHeight*outWidth), dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchPool2DRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			1,
			1,
			uint32(inHeight),
			uint32(inWidth),
			uint32(outHeight),
			uint32(outWidth),
			true,
			false,
		)
	}

	harness.Sync()
}

func poolDims(count int, config cpupool.PoolConfig) (int, int, int, int) {
	inSide := config.KernelH + config.StrideH*(poolOutputSide(count)-1)

	if inSide < config.KernelH {
		inSide = config.KernelH
	}

	outSide := (inSide-config.KernelH)/config.StrideH + 1

	return inSide, inSide, outSide, outSide
}

func poolOutputSide(count int) int {
	side := 1

	for side*side < count {
		side++
	}

	return side
}
