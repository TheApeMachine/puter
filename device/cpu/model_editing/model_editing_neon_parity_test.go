//go:build arm64

package model_editing

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestWeightGraftAddFloat32NEONParity(t *testing.T) {
	convey.Convey("Given WeightGraftAddFloat32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WeightGraftAddFloat32Scalar for N=%d", length), func() {
				weights, injection := randomGraftVectors(length, 0x2B50+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				WeightGraftAddFloat32NEON(&got[0], &injection[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				weights, injection := randomGraftVectors(length, 0x2B51+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				WeightGraftAddFloat32NEONAsm(&got[0], &injection[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
