//go:build amd64

package dequant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2DequantAvailable() bool {
	return cpu.X86.HasAVX2
}

func sse2DequantAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestDequantInt8AVX2Parity(t *testing.T) {
	if !avx2DequantAvailable() {
		t.Skip("AVX2 required")
	}

	convey.Convey("Given DequantInt8AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt8Generic for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x4020 + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				dequantInt8AVX2(got, source, scale, zeroPoint)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match dequantInt8Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				rng := rand.New(rand.NewSource(0x4021 + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				DequantInt8AVX2Asm(
					&got[0], &source[0], length,
					scale, int16(zeroPoint),
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestDequantInt8SSE2Parity(t *testing.T) {
	if !sse2DequantAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given DequantInt8SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt8Generic for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x4030 + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				dequantInt8SSE2(got, source, scale, zeroPoint)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match dequantInt8Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				rng := rand.New(rand.NewSource(0x4031 + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				DequantInt8SSE2Asm(
					&got[0], &source[0], length,
					scale, int16(zeroPoint),
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
