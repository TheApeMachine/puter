//go:build xla

package convolution_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuconvolution "github.com/theapemachine/puter/device/cpu/convolution"
	"github.com/theapemachine/puter/device/xla"
	"github.com/theapemachine/puter/device/xla/internal/hlo"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceConvolution = cpuconvolution.New()

func TestConv2DXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	config := device.Conv2DConfig{
		StrideH: 1, StrideW: 1,
		PaddingH: 1, PaddingW: 1,
		DilationH: 1, DilationW: 1,
	}

	convey.Convey("Given XLA Conv2D", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, side := range []int{4, 7, 8} {
					convey.Convey(fmt.Sprintf("side=%d", side), func() {
						runConv2DParityCase(
							t, harness, config, storageDType,
							1, 2, side, side,
							3, 3, 3,
						)
					})
				}
			})
		}
	})
}

func TestConv1DXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	config := device.Conv1DConfig{Stride: 1, Padding: 1, Dilation: 1}

	convey.Convey("Given XLA Conv1D", t, func() {
		for _, length := range []int{4, 7, 64} {
			convey.Convey(fmt.Sprintf("length=%d", length), func() {
				runConv1DParityCase(t, harness, config, dtype.Float32, 1, 2, length, 3, 3)
			})
		}
	})
}

func TestConv3DXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	config := device.Conv3DConfig{
		StrideD: 1, StrideH: 1, StrideW: 1,
		PaddingD: 0, PaddingH: 0, PaddingW: 0,
		DilationD: 1, DilationH: 1, DilationW: 1,
	}

	convey.Convey("Given XLA Conv3D", t, func() {
		runConv3DParityCase(
			t, harness, config, dtype.Float32,
			1, 2, 3, 4, 4,
			3, 2, 2, 2,
		)
	})
}

func TestConvTranspose2DXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	config := device.Conv2DConfig{
		StrideH: 2, StrideW: 2,
		PaddingH: 0, PaddingW: 0,
		DilationH: 1, DilationW: 1,
	}

	convey.Convey("Given XLA ConvTranspose2D", t, func() {
		runConvTranspose2DParityCase(
			t, harness, config, dtype.Float32,
			1, 2, 4, 4,
			2, 3, 3,
			8, 8,
		)
	})
}

func runConv2DParityCase(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	config device.Conv2DConfig,
	format dtype.DType,
	batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth int,
) {
	outHeight := hlo.ConvOutputSize(inHeight, kernelHeight, config.PaddingH, config.StrideH, config.DilationH)
	outWidth := hlo.ConvOutputSize(inWidth, kernelWidth, config.PaddingW, config.StrideW, config.DilationW)

	input := xlaparity.RandomUnaryInput(batch*inChannels*inHeight*inWidth, 0x6100)
	weight := xlaparity.RandomUnaryInput(outChannels*inChannels*kernelHeight*kernelWidth, 0x6200)
	bias := xlaparity.RandomUnaryInput(outChannels, 0x6300)
	want := make([]float32, batch*outChannels*outHeight*outWidth)

	referenceConvolution.Conv2D(
		config,
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&weight[0]),
		unsafe.Pointer(&bias[0]),
		unsafe.Pointer(&want[0]),
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	inputTensor := harness.UploadVector(input, format)
	weightTensor := harness.UploadVector(weight, format)
	biasTensor := harness.UploadVector(bias, format)
	outputTensor := harness.UploadVector(make([]float32, len(want)), format)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	harness.Backend().Conv2D(
		config,
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(weightTensor),
		xla.ResidentPointer(biasTensor),
		xla.ResidentPointer(outputTensor),
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	got := harness.DownloadFloat32(outputTensor, format)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func runConv1DParityCase(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	config device.Conv1DConfig,
	format dtype.DType,
	batch, inChannels, inLength, outChannels, kernelLength int,
) {
	outLength := hlo.ConvOutputSize(inLength, kernelLength, config.Padding, config.Stride, config.Dilation)

	input := xlaparity.RandomUnaryInput(batch*inChannels*inLength, 0x6400)
	weight := xlaparity.RandomUnaryInput(outChannels*inChannels*kernelLength, 0x6500)
	bias := xlaparity.RandomUnaryInput(outChannels, 0x6600)
	want := make([]float32, batch*outChannels*outLength)

	referenceConvolution.Conv1D(
		config,
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&weight[0]),
		unsafe.Pointer(&bias[0]),
		unsafe.Pointer(&want[0]),
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
	)

	inputTensor := harness.UploadVector(input, format)
	weightTensor := harness.UploadVector(weight, format)
	biasTensor := harness.UploadVector(bias, format)
	outputTensor := harness.UploadVector(make([]float32, len(want)), format)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	harness.Backend().Conv1D(
		config,
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(weightTensor),
		xla.ResidentPointer(biasTensor),
		xla.ResidentPointer(outputTensor),
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
	)

	got := harness.DownloadFloat32(outputTensor, format)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func runConv3DParityCase(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	config device.Conv3DConfig,
	format dtype.DType,
	batch, inChannels, inDepth, inHeight, inWidth,
	outChannels, kernelDepth, kernelHeight, kernelWidth int,
) {
	outDepth := hlo.ConvOutputSize(inDepth, kernelDepth, config.PaddingD, config.StrideD, config.DilationD)
	outHeight := hlo.ConvOutputSize(inHeight, kernelHeight, config.PaddingH, config.StrideH, config.DilationH)
	outWidth := hlo.ConvOutputSize(inWidth, kernelWidth, config.PaddingW, config.StrideW, config.DilationW)

	inputCount := batch * inChannels * inDepth * inHeight * inWidth
	weightCount := outChannels * inChannels * kernelDepth * kernelHeight * kernelWidth
	outputCount := batch * outChannels * outDepth * outHeight * outWidth

	input := xlaparity.RandomUnaryInput(inputCount, 0x6700)
	weight := xlaparity.RandomUnaryInput(weightCount, 0x6800)
	bias := xlaparity.RandomUnaryInput(outChannels, 0x6900)
	want := make([]float32, outputCount)

	referenceConvolution.Conv3D(
		config,
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&weight[0]),
		unsafe.Pointer(&bias[0]),
		unsafe.Pointer(&want[0]),
		batch, inChannels, inDepth, inHeight, inWidth,
		outChannels, kernelDepth, kernelHeight, kernelWidth,
		outDepth, outHeight, outWidth,
		format,
	)

	inputTensor := harness.UploadVector(input, format)
	weightTensor := harness.UploadVector(weight, format)
	biasTensor := harness.UploadVector(bias, format)
	outputTensor := harness.UploadVector(make([]float32, outputCount), format)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	harness.Backend().Conv3D(
		config,
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(weightTensor),
		xla.ResidentPointer(biasTensor),
		xla.ResidentPointer(outputTensor),
		batch, inChannels, inDepth, inHeight, inWidth,
		outChannels, kernelDepth, kernelHeight, kernelWidth,
		outDepth, outHeight, outWidth,
		format,
	)

	got := harness.DownloadFloat32(outputTensor, format)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func runConvTranspose2DParityCase(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	config device.Conv2DConfig,
	format dtype.DType,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth, outHeight, outWidth int,
) {
	input := xlaparity.RandomUnaryInput(batch*inChannels*inHeight*inWidth, 0x6A00)
	weight := xlaparity.RandomUnaryInput(inChannels*outChannels*kernelHeight*kernelWidth, 0x6B00)
	bias := xlaparity.RandomUnaryInput(outChannels, 0x6C00)
	want := make([]float32, batch*outChannels*outHeight*outWidth)

	referenceConvolution.ConvTranspose2D(
		config,
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&weight[0]),
		unsafe.Pointer(&bias[0]),
		unsafe.Pointer(&want[0]),
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	inputTensor := harness.UploadVector(input, format)
	weightTensor := harness.UploadVector(weight, format)
	biasTensor := harness.UploadVector(bias, format)
	outputTensor := harness.UploadVector(make([]float32, len(want)), format)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	harness.Backend().ConvTranspose2D(
		config,
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(weightTensor),
		xla.ResidentPointer(biasTensor),
		xla.ResidentPointer(outputTensor),
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	got := harness.DownloadFloat32(outputTensor, format)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}
