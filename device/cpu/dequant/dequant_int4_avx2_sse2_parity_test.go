//go:build amd64

package dequant

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestDequantInt4AVX2Parity(t *testing.T) {
	if !avx2DequantAvailable() {
		t.Skip("AVX2 required")
	}

	convey.Convey("Given DequantInt4AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt4Generic for N=%d", length), func() {
				bytes := int4BytesFromLength(length, 0x4040+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				dequantInt4AVX2(got, pairs.Bytes(), length, scale, zeroPoint)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match dequantInt4Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				bytes := int4BytesFromLength(length, 0x4041+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				DequantInt4AVX2Asm(
					&got[0], &bytes[0], length,
					scale, zeroPoint,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestDequantInt4SSE2Parity(t *testing.T) {
	if !sse2DequantAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given DequantInt4SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt4Generic for N=%d", length), func() {
				bytes := int4BytesFromLength(length, 0x4050+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				dequantInt4SSE2(got, pairs.Bytes(), length, scale, zeroPoint)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match dequantInt4Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				bytes := int4BytesFromLength(length, 0x4051+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				DequantInt4SSE2Asm(
					&got[0], &bytes[0], length,
					scale, zeroPoint,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
