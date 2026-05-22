//go:build arm64

package tokenizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestTokenizerPackInt32NEONParity(t *testing.T) {
	convey.Convey("Given TokenizerPackInt32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PackInt32Scalar for N=%d", length), func() {
				source := randomInt32Vector(length, 0x2830+int64(length))
				want := make([]int32, length)
				got := make([]int32, length)

				PackInt32Scalar(want, source)
				TokenizerPackInt32NEON(&got[0], &source[0], length)

				assertInt32SliceEqual(t, got, want)
			})
		}

		convey.Convey("It should match PackInt32Scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomInt32Vector(length, 0x2831+int64(length))
				want := make([]int32, length)
				got := make([]int32, length)

				PackInt32Scalar(want, source)
				TokenizerPackInt32NEONAsm(&got[0], &source[0], length)

				assertInt32SliceEqual(t, got, want)
			}
		})
	})
}
