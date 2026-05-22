//go:build amd64

package model_editing

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2ModelEditingAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2ModelEditingAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestWeightGraftAddFloat32AVX2Parity(t *testing.T) {
	if !avx2ModelEditingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runWeightGraftAddParity(t, WeightGraftAddFloat32AVX2, WeightGraftAddFloat32AVX2Asm, 0x3B40)
}

func TestWeightGraftAddFloat32SSE2Parity(t *testing.T) {
	if !sse2ModelEditingAvailable() {
		t.Skip("SSE2 required")
	}

	runWeightGraftAddParity(t, WeightGraftAddFloat32SSE2, WeightGraftAddFloat32SSE2Asm, 0x3B50)
}

func runWeightGraftAddParity(
	testingObject *testing.T,
	runWrapper func(*float32, *float32, int),
	runAsm func(*float32, *float32, int),
	seedBase int64,
) {
	convey.Convey("Given weight graft add SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WeightGraftAddFloat32Scalar for N=%d", length), func() {
				weights, injection := randomGraftVectors(length, seedBase+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				runWrapper(&got[0], &injection[0], length)

				assertFloat32SliceEqual(testingObject, got, want)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				weights, injection := randomGraftVectors(length, seedBase+0x100+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				runAsm(&got[0], &injection[0], length)

				assertFloat32SliceEqual(testingObject, got, want)
			}
		})
	})
}
