//go:build xla

package xla_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	cpumatmul "github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
	"github.com/theapemachine/puter/fusion"
)

var (
	referenceMatmul     = cpumatmul.New()
	referenceActivation = cpuactivation.New()
	referenceLayerNorm  = cpulayernorm.New()
)

func TestFusionCatalogXLACoverage(t *testing.T) {
	convey.Convey("Given the fusion catalog", t, func() {
		entry := fusion.Default.Lookup(
			[]string{"matmul", "add", "gelu"},
			[]dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			tensor.LayoutDense,
			tensor.XLA,
		)

		convey.Convey("It should expose matmul_bias_gelu for XLA", func() {
			convey.So(entry, convey.ShouldNotBeNil)
			convey.So(entry.FusedOp, convey.ShouldEqual, "matmul_bias_gelu")
		})
	})
}

func TestMatmulBiasGeluXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA matmul+bias+gelu fusion", t, func() {
		for _, inner := range []int{1, 7, 64, 1024, 8192} {
			convey.Convey(fmt.Sprintf("K=%d", inner), func() {
				const rows = 8
				const cols = 8

				left := xlaparity.RandomUnaryInput(rows*inner, 0xf100+int64(inner))
				right := xlaparity.RandomUnaryInput(inner*cols, 0xf200+int64(inner))
				bias := xlaparity.RandomUnaryInput(cols, 0xf300+int64(inner))
				want := matmulBiasGeluReference(left, right, bias, rows, inner, cols)

				leftTensor := harness.UploadMatrix(left, rows, inner, dtype.Float32)
				rightTensor := harness.UploadMatrix(right, inner, cols, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, rows*cols), rows, cols, dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer biasTensor.Close()
				defer outputTensor.Close()

				metricsBefore := harness.Backend().BuilderCacheMetrics()

				harness.Backend().MatmulBiasGelu(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(leftTensor),
					xla.ResidentPointer(rightTensor),
					xla.ResidentPointer(biasTensor),
					rows, inner, cols,
					dtype.Float32,
				)

				metricsAfter := harness.Backend().BuilderCacheMetrics()
				convey.So(metricsAfter.Executes, convey.ShouldEqual, metricsBefore.Executes+1)
				convey.So(metricsAfter.Compiles, convey.ShouldEqual, metricsBefore.Compiles+1)

				harness.Backend().MatmulBiasGelu(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(leftTensor),
					xla.ResidentPointer(rightTensor),
					xla.ResidentPointer(biasTensor),
					rows, inner, cols,
					dtype.Float32,
				)

				metricsCached := harness.Backend().BuilderCacheMetrics()
				convey.So(metricsCached.Executes, convey.ShouldEqual, metricsAfter.Executes+1)
				convey.So(metricsCached.Compiles, convey.ShouldEqual, metricsAfter.Compiles)
				convey.So(metricsCached.Hits, convey.ShouldEqual, metricsAfter.Hits+1)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}

func TestLayernormResidualXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA layernorm+residual fusion", t, func() {
		for _, lastDim := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", lastDim), func() {
				const rows = 4

				input := xlaparity.RandomUnaryInput(rows*lastDim, 0xf400+int64(lastDim))
				scale := xlaparity.RandomUnaryInput(lastDim, 0xf500+int64(lastDim))
				bias := xlaparity.RandomUnaryInput(lastDim, 0xf600+int64(lastDim))
				residual := xlaparity.RandomUnaryInput(rows*lastDim, 0xf700+int64(lastDim))
				want := layernormResidualReference(input, scale, bias, residual, rows, lastDim)

				inputTensor := harness.UploadMatrix(input, rows, lastDim, dtype.Float32)
				scaleTensor := harness.UploadVector(scale, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				residualTensor := harness.UploadMatrix(residual, rows, lastDim, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, rows*lastDim), rows, lastDim, dtype.Float32)
				defer inputTensor.Close()
				defer scaleTensor.Close()
				defer biasTensor.Close()
				defer residualTensor.Close()
				defer outputTensor.Close()

				harness.Backend().LayernormResidual(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(scaleTensor),
					xla.ResidentPointer(biasTensor),
					xla.ResidentPointer(residualTensor),
					rows, lastDim,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}

func matmulBiasGeluReference(
	left, right, bias []float32,
	rows, inner, cols int,
) []float32 {
	matmulOut := make([]float32, rows*cols)
	referenceMatmul.Matmul(
		unsafe.Pointer(&matmulOut[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		rows, inner, cols,
		dtype.Float32,
	)

	biased := make([]float32, rows*cols)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			offset := rowIndex*cols + colIndex
			biased[offset] = matmulOut[offset] + bias[colIndex]
		}
	}

	want := make([]float32, rows*cols)
	referenceActivation.Gelu(
		unsafe.Pointer(&want[0]),
		unsafe.Pointer(&biased[0]),
		rows*cols,
		dtype.Float32,
	)

	return want
}

func layernormResidualReference(
	input, scale, bias, residual []float32,
	rows, lastDim int,
) []float32 {
	normed := make([]float32, rows*lastDim)
	referenceLayerNorm.LayerNorm(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&scale[0]),
		unsafe.Pointer(&bias[0]),
		unsafe.Pointer(&normed[0]),
		rows, lastDim,
		dtype.Float32,
	)

	want := make([]float32, rows*lastDim)

	for index := range want {
		want[index] = normed[index] + residual[index]
	}

	return want
}

func BenchmarkMatmulBiasGeluXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	const rows = 8
	const inner = 8192
	const cols = 8

	left := xlaparity.RandomUnaryInput(rows*inner, 0xf800)
	right := xlaparity.RandomUnaryInput(inner*cols, 0xf810)
	bias := xlaparity.RandomUnaryInput(cols, 0xf820)
	leftTensor := harness.UploadMatrix(left, rows, inner, dtype.Float32)
	rightTensor := harness.UploadMatrix(right, inner, cols, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadMatrix(make([]float32, rows*cols), rows, cols, dtype.Float32)
	defer leftTensor.Close()
	defer rightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().MatmulBiasGelu(
			xla.ResidentPointer(outputTensor),
			xla.ResidentPointer(leftTensor),
			xla.ResidentPointer(rightTensor),
			xla.ResidentPointer(biasTensor),
			rows, inner, cols,
			dtype.Float32,
		)
	}
}
