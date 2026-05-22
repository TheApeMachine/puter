//go:build arm64

package interpretability

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestActivationSteerFloat32NEONParity(t *testing.T) {
	convey.Convey("Given ActivationSteerFloat32NEON", t, func() {
		const coefficient = float32(0.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ActivationSteerFloat32Scalar for N=%d", length), func() {
				base, direction := randomSteerVectors(length, 0x2A50+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				ActivationSteerFloat32NEON(&got[0], &base[0], &direction[0], coefficient, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				base, direction := randomSteerVectors(length, 0x2A51+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				ActivationSteerFloat32NEONAsm(&got[0], &base[0], &direction[0], coefficient, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
