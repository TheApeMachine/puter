//go:build amd64

package interpretability

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2InterpretabilityAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2InterpretabilityAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestActivationSteerFloat32AVX2Parity(t *testing.T) {
	if !avx2InterpretabilityAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runActivationSteerParity(t, ActivationSteerFloat32AVX2, ActivationSteerFloat32AVX2Asm, 0x3A40)
}

func TestActivationSteerFloat32SSE2Parity(t *testing.T) {
	if !sse2InterpretabilityAvailable() {
		t.Skip("SSE2 required")
	}

	runActivationSteerParity(t, ActivationSteerFloat32SSE2, ActivationSteerFloat32SSE2Asm, 0x3A50)
}

func runActivationSteerParity(
	testingObject *testing.T,
	runWrapper func(*float32, *float32, *float32, float32, int),
	runAsm func(*float32, *float32, *float32, float32, int),
	seedBase int64,
) {
	convey.Convey("Given activation steer SIMD kernel", testingObject, func() {
		const coefficient = float32(0.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ActivationSteerFloat32Scalar for N=%d", length), func() {
				base, direction := randomSteerVectors(length, seedBase+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				runWrapper(&got[0], &base[0], &direction[0], coefficient, length)

				assertFloat32SliceEqual(testingObject, got, want)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				base, direction := randomSteerVectors(length, seedBase+0x100+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				runAsm(&got[0], &base[0], &direction[0], coefficient, length)

				assertFloat32SliceEqual(testingObject, got, want)
			}
		})
	})
}
