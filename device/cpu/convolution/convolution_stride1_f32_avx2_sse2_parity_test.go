//go:build amd64

package convolution

import (
	"math/rand"
	"testing"

	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func conv2DStride1RowF32Reference(
	outRow []float32,
	input, weight []float32,
	biasValue float32,
	outCols, inChannels, kH, kW,
	inHStride, inCStride, wHStride, wCStride,
	ihStart int,
) {
	for outCol := 0; outCol < outCols; outCol++ {
		sum := biasValue

		for inChIndex := 0; inChIndex < inChannels; inChIndex++ {
			inputChannelOffset := inChIndex * inCStride
			weightChannelOffset := inChIndex * wCStride

			for khIndex := 0; khIndex < kH; khIndex++ {
				inputRowOffset := (ihStart + khIndex) * inHStride
				weightRowOffset := khIndex * wHStride

				for kwIndex := 0; kwIndex < kW; kwIndex++ {
					inputIndex := inputChannelOffset + inputRowOffset + outCol + kwIndex
					weightIndex := weightChannelOffset + weightRowOffset + kwIndex
					sum += input[inputIndex] * weight[weightIndex]
				}
			}
		}

		outRow[outCol] = sum
	}
}

func testConv2DStride1RowF32Parity(
	t *testing.T,
	kernel func(
		outRow, input, weight *float32,
		biasValue float32,
		outCols, inChannels, kH, kW int,
		inHStride, inCStride, wHStride, wCStride int,
		ihStart, iwStart int,
	),
) {
	t.Helper()

	const (
		inC  = 3
		inH  = 8
		inW  = 8
		kH   = 3
		kW   = 3
		outH = inH - kH + 1
		outW = inW - kW + 1
	)

	rng := rand.New(rand.NewSource(0xC033))
	input := make([]float32, inC*inH*inW)
	weight := make([]float32, inC*kH*kW)

	for index := range input {
		input[index] = float32(rng.NormFloat64())
	}

	for index := range weight {
		weight[index] = float32(rng.NormFloat64())
	}

	biasValue := float32(rng.NormFloat64())
	blockCols := outW &^ 3

	want := make([]float32, blockCols)
	conv2DStride1RowF32Reference(
		want,
		input, weight,
		biasValue,
		blockCols,
		inC, kH, kW,
		inW, inH*inW,
		kW, kH*kW,
		0,
	)

	got := make([]float32, blockCols)
	kernel(
		&got[0],
		&input[0],
		&weight[0],
		biasValue,
		blockCols,
		inC, kH, kW,
		inW, inH*inW,
		kW, kH*kW,
		0, 0,
	)

	parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
}

func TestConv2DStride1RowF32AVX2Parity(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skip("AVX2+FMA required")
	}

	testConv2DStride1RowF32Parity(t, Conv2dStride1RowF32AVX2Asm)
}

func TestConv2DStride1RowF32SSE2Parity(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 required")
	}

	testConv2DStride1RowF32Parity(t, Conv2dStride1RowF32SSE2Asm)
}
