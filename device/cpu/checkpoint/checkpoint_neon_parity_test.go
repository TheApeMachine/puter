//go:build arm64

package checkpoint

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestCheckpointEncodeFloat32DataNEONParity(t *testing.T) {
	convey.Convey("Given CheckpointEncodeFloat32DataNEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match encodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2960+int64(length))
				want := make([]uint8, length*4)
				got := make([]uint8, length*4)

				encodeFloat32DataScalar(want, source)
				CheckpointEncodeFloat32DataNEON(&got[0], &source[0], length)

				assertUint8PayloadEqual(t, got, want)
			})
		}

		convey.Convey("It should match encodeFloat32DataScalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomFloat32Vector(length, 0x2961+int64(length))
				want := make([]uint8, length*4)
				got := make([]uint8, length*4)

				encodeFloat32DataScalar(want, source)
				CheckpointEncodeFloat32DataNEONAsm(&got[0], &source[0], length)

				assertUint8PayloadEqual(t, got, want)
			}
		})
	})
}

func TestCheckpointDecodeFloat32DataNEONParity(t *testing.T) {
	convey.Convey("Given CheckpointDecodeFloat32DataNEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match decodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2970+int64(length))
				payload := make([]uint8, length*4)
				encodeFloat32DataScalar(payload, source)

				want := make([]float32, length)
				got := make([]float32, length)

				decodeFloat32DataScalar(want, payload)
				CheckpointDecodeFloat32DataNEON(&got[0], &payload[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match decodeFloat32DataScalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomFloat32Vector(length, 0x2971+int64(length))
				payload := make([]uint8, length*4)
				encodeFloat32DataScalar(payload, source)

				want := make([]float32, length)
				got := make([]float32, length)

				decodeFloat32DataScalar(want, payload)
				CheckpointDecodeFloat32DataNEONAsm(&got[0], &payload[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
