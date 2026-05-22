//go:build arm64

package convolution

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestConv2DStride1RowBF16NEONAsm(t *testing.T) {
	const (
		inC  = 3
		inH  = 8
		inW  = 8
		kH   = 3
		kW   = 3
		outH = inH - kH + 1
		outW = inW - kW + 1
	)

	config := DefaultConv2DConfig()
	rng := rand.New(rand.NewSource(0xBF16))

	input := make([]dtype.BF16, inC*inH*inW)
	weight := make([]dtype.BF16, inC*kH*kW)

	for index := range input {
		input[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
	}

	for index := range weight {
		weight[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
	}

	bias := dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
	biasValue := (&bias).Float32()

	scalar := make([]dtype.BF16, outW)
	loadInput, _ := elementAccessors(dtype.BFloat16)
	loadWeight, _ := elementAccessors(dtype.BFloat16)

	for outCol := range outW {
		scalar[outCol] = dtype.NewBfloat16FromFloat32(conv2DPixelTyped(
			config,
			unsafe.Pointer(unsafe.SliceData(input)),
			unsafe.Pointer(unsafe.SliceData(weight)),
			loadInput, loadWeight,
			0, 0,
			inC, inH, inW,
			kH, kW,
			0, outCol,
			biasValue,
		))
	}

	got := make([]dtype.BF16, 4)
	Conv2dStride1RowBF16NEONAsm(
		(*uint16)(unsafe.Pointer(&got[0])),
		(*uint16)(unsafe.Pointer(&input[0])),
		(*uint16)(unsafe.Pointer(&weight[0])),
		biasValue,
		4,
		inC, kH, kW,
		inW, inH*inW,
		kW, kH*kW,
		0, 0,
	)

	for index := range 4 {
		expectedBits := uint16(scalar[index])
		actualBits := uint16(got[index])

		if expectedBits == actualBits {
			continue
		}

		diff := int(expectedBits) - int(actualBits)
		if diff < 0 {
			diff = -diff
		}

		if diff > 2 {
			t.Fatalf("lane %d expected=0x%04x actual=0x%04x ulp=%d",
				index, expectedBits, actualBits, diff)
		}
	}
}

func TestConv2DPatchDotBF16NEONAsm(t *testing.T) {
	for _, count := range parity.Lengths {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			weight := make([]dtype.BF16, count)
			patch := make([]dtype.BF16, count)
			rng := rand.New(rand.NewSource(int64(count) + 0xBF16D00))

			for index := range count {
				weight[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
				patch[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
			}

			var scalar float32

			for index := range count {
				leftValue := (&weight[index]).Float32()
				rightValue := (&patch[index]).Float32()
				scalar += leftValue * rightValue
			}

			got := Conv2dPatchDotBF16NEONAsm(
				(*uint16)(&weight[0]),
				(*uint16)(&patch[0]),
				count,
			)

			scalarBits := uint16(dtype.NewBfloat16FromFloat32(scalar))
			gotBits := uint16(dtype.NewBfloat16FromFloat32(got))

			diff := int(scalarBits) - int(gotBits)
			if diff < 0 {
				diff = -diff
			}

			if diff > 2 {
				t.Fatalf("N=%d scalar=0x%04x (%g) neon=0x%04x (%g) ulp_bf16=%d",
					count, scalarBits, scalar, gotBits, got, diff)
			}
		})
	}
}
