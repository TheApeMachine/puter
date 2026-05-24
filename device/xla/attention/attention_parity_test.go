//go:build xla

package attention_test

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuattention "github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestAttentionXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA scaled dot product attention", t, func() {
		seqQ := 4
		seqK := 5
		depth := 8
		valueDim := 8

		query := xlaparity.RandomUnaryInput(seqQ*depth, 0x5100)
		key := xlaparity.RandomUnaryInput(seqK*depth, 0x5200)
		value := xlaparity.RandomUnaryInput(seqK*valueDim, 0x5300)
		want := make([]float32, seqQ*valueDim)

		cpuattention.ScaledDotProductAttention(
			cpuattention.DefaultFlashAttentionConfig(),
			&query[0], &key[0], &value[0], &want[0],
			seqQ, seqK, depth, valueDim,
			dtype.Float32,
		)

		queryTensor := harness.UploadMatrix(query, seqQ, depth, dtype.Float32)
		keyTensor := harness.UploadMatrix(key, seqK, depth, dtype.Float32)
		valueTensor := harness.UploadMatrix(value, seqK, valueDim, dtype.Float32)
		outputTensor := harness.UploadMatrix(make([]float32, seqQ*valueDim), seqQ, valueDim, dtype.Float32)
		defer queryTensor.Close()
		defer keyTensor.Close()
		defer valueTensor.Close()
		defer outputTensor.Close()

		harness.Backend().ScaledDotProductAttention(
			device.FlashAttentionConfig{Causal: false},
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(keyTensor),
			xla.ResidentPointer(valueTensor),
			xla.ResidentPointer(outputTensor),
			seqQ, seqK, depth, valueDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA causal scaled dot product attention", t, func() {
		seqQ := 4
		seqK := 4
		depth := 8
		valueDim := 8

		query := xlaparity.RandomUnaryInput(seqQ*depth, 0x5400)
		key := xlaparity.RandomUnaryInput(seqK*depth, 0x5500)
		value := xlaparity.RandomUnaryInput(seqK*valueDim, 0x5600)
		want := make([]float32, seqQ*valueDim)

		cpuattention.ScaledDotProductAttention(
			device.FlashAttentionConfig{Causal: true},
			&query[0], &key[0], &value[0], &want[0],
			seqQ, seqK, depth, valueDim,
			dtype.Float32,
		)

		queryTensor := harness.UploadMatrix(query, seqQ, depth, dtype.Float32)
		keyTensor := harness.UploadMatrix(key, seqK, depth, dtype.Float32)
		valueTensor := harness.UploadMatrix(value, seqK, valueDim, dtype.Float32)
		outputTensor := harness.UploadMatrix(make([]float32, seqQ*valueDim), seqQ, valueDim, dtype.Float32)
		defer queryTensor.Close()
		defer keyTensor.Close()
		defer valueTensor.Close()
		defer outputTensor.Close()

		harness.Backend().ScaledDotProductAttention(
			device.FlashAttentionConfig{Causal: true},
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(keyTensor),
			xla.ResidentPointer(valueTensor),
			xla.ResidentPointer(outputTensor),
			seqQ, seqK, depth, valueDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA multi head attention", t, func() {
		seqQ := 3
		seqK := 4
		numHeads := 2
		headDim := 8
		config := device.MultiHeadAttentionConfig{
			NumHeads: numHeads,
			HeadDim:  headDim,
			Causal:   true,
		}

		query := xlaparity.RandomUnaryInput(seqQ*numHeads*headDim, 0x5700)
		key := xlaparity.RandomUnaryInput(seqK*numHeads*headDim, 0x5800)
		value := xlaparity.RandomUnaryInput(seqK*numHeads*headDim, 0x5900)
		want := make([]float32, seqQ*numHeads*headDim)

		cpuattention.MultiHeadAttention(
			config,
			&query[0], &key[0], &value[0], &want[0],
			seqQ, seqK,
			dtype.Float32,
		)

		queryTensor := harness.UploadMatrix(query, seqQ, numHeads*headDim, dtype.Float32)
		keyTensor := harness.UploadMatrix(key, seqK, numHeads*headDim, dtype.Float32)
		valueTensor := harness.UploadMatrix(value, seqK, numHeads*headDim, dtype.Float32)
		outputTensor := harness.UploadMatrix(make([]float32, seqQ*numHeads*headDim), seqQ, numHeads*headDim, dtype.Float32)
		defer queryTensor.Close()
		defer keyTensor.Close()
		defer valueTensor.Close()
		defer outputTensor.Close()

		harness.Backend().MultiHeadAttention(
			config,
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(keyTensor),
			xla.ResidentPointer(valueTensor),
			xla.ResidentPointer(outputTensor),
			seqQ, seqK,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA grouped query attention", t, func() {
		seqQ := 2
		seqK := 3
		numHeads := 4
		kvHeads := 2
		headDim := 8
		config := device.MultiHeadAttentionConfig{
			NumHeads:    numHeads,
			HeadDim:     headDim,
			KVHeadCount: kvHeads,
		}

		query := xlaparity.RandomUnaryInput(seqQ*numHeads*headDim, 0x5A00)
		key := xlaparity.RandomUnaryInput(seqK*kvHeads*headDim, 0x5B00)
		value := xlaparity.RandomUnaryInput(seqK*kvHeads*headDim, 0x5C00)
		want := make([]float32, seqQ*numHeads*headDim)

		cpuattention.MultiHeadAttention(
			config,
			&query[0], &key[0], &value[0], &want[0],
			seqQ, seqK,
			dtype.Float32,
		)

		queryTensor := harness.UploadMatrix(query, seqQ, numHeads*headDim, dtype.Float32)
		keyTensor := harness.UploadMatrix(key, seqK, kvHeads*headDim, dtype.Float32)
		valueTensor := harness.UploadMatrix(value, seqK, kvHeads*headDim, dtype.Float32)
		outputTensor := harness.UploadMatrix(make([]float32, seqQ*numHeads*headDim), seqQ, numHeads*headDim, dtype.Float32)
		defer queryTensor.Close()
		defer keyTensor.Close()
		defer valueTensor.Close()
		defer outputTensor.Close()

		harness.Backend().MultiHeadAttention(
			config,
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(keyTensor),
			xla.ResidentPointer(valueTensor),
			xla.ResidentPointer(outputTensor),
			seqQ, seqK,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})
}
