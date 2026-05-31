//go:build darwin && cgo

package parity

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

/*
AssertDecodedSlicesMatch compares Metal download results against a reference.
Float32 lanes use float32 ULP distance. Reduced-precision dtypes compare the
encoded storage bit patterns produced by rounding each decoded lane back to the
storage format, which matches how parity references are built.
*/
func AssertDecodedSlicesMatch(
	testingTB testing.TB,
	got, want []float32,
	format dtype.DType,
	maxULP int,
) {
	testingTB.Helper()

	if format == dtype.Float32 {
		cpuparity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
		return
	}

	if len(got) != len(want) {
		testingTB.Fatal("length mismatch got=", len(got), " want=", len(want))
	}

	for index := range got {
		gotBits := storageBits(got[index], format)
		wantBits := storageBits(want[index], format)
		bitDistance := storageBitDistance(gotBits, wantBits)

		if bitDistance <= maxULP {
			continue
		}

		testingTB.Fatal(
			"lane ", index,
			" got=", got[index],
			" want=", want[index],
			" storage_bits=", gotBits,
			" want_bits=", wantBits,
			" bit_distance=", bitDistance,
			" max=", maxULP,
		)
	}
}

func storageBits(value float32, format dtype.DType) uint16 {
	switch format {
	case dtype.Float16:
		return uint16(dtype.Fromfloat32(value).Bits())
	case dtype.BFloat16:
		return uint16(dtype.NewBfloat16FromFloat32(value).Bits())
	default:
		return 0
	}
}

func storageBitDistance(leftBits, rightBits uint16) int {
	left := int(leftBits)
	right := int(rightBits)

	if left > right {
		left, right = right, left
	}

	return right - left
}
