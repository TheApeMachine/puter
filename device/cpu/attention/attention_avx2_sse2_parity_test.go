//go:build amd64

package attention

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const flashAttentionReducedMaxULP = 0

func avx2AttentionAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2AttentionAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestFlashAttentionOnlineUpdateAVX2Parity(t *testing.T) {
	if !avx2AttentionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given FlashAttentionOnlineUpdateAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA80+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA81+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.37)
				shifted := float32(0.91)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				flashAttentionOnlineUpdateAVX2(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionReducedMaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA82+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA83+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.41)
				shifted := float32(0.83)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				FlashAttentionOnlineUpdateAVX2Asm(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionReducedMaxULP,
				)
			}
		})
	})
}

func TestFlashAttentionScaleAVX2Parity(t *testing.T) {
	if !avx2AttentionAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given FlashAttentionScaleAVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA84+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.125)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				flashAttentionScaleAVX2(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionReducedMaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA85+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.25)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				FlashAttentionScaleAVX2Asm(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionReducedMaxULP,
				)
			}
		})
	})
}

func TestFlashAttentionOnlineUpdateSSE2Parity(t *testing.T) {
	if !sse2AttentionAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given FlashAttentionOnlineUpdateSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA90+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA91+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.37)
				shifted := float32(0.91)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				flashAttentionOnlineUpdateSSE2(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionReducedMaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA92+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA93+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.41)
				shifted := float32(0.83)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				FlashAttentionOnlineUpdateSSE2Asm(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionReducedMaxULP,
				)
			}
		})
	})
}

func TestFlashAttentionScaleSSE2Parity(t *testing.T) {
	if !sse2AttentionAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given FlashAttentionScaleSSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA94+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.125)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				flashAttentionScaleSSE2(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionReducedMaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA95+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.25)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				FlashAttentionScaleSSE2Asm(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionReducedMaxULP,
				)
			}
		})
	})
}
