//go:build arm64

package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const shapeNEONMaxULP = 0

func TestCopyContiguousF32NEONParity(t *testing.T) {
	convey.Convey("Given CopyContiguousF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CopyContiguousGeneric for N=%d", length), func() {
				source := randomShapeFloat32(length, 0x1840+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousF32NEON(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			})
		}

		convey.Convey("It should match CopyContiguousGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomShapeFloat32(length, 0x1841+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousFloat32NEONAsm(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			}
		})
	})
}

func TestWhereF32NEONParity(t *testing.T) {
	convey.Convey("Given WhereF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WhereGeneric for N=%d", length), func() {
				positive := randomShapeFloat32(length, 0x1850+int64(length))
				negative := randomShapeFloat32(length, 0x1851+int64(length))
				mask := shapeMaskBytes(length, 0x1852+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereF32NEON(&got[0], &positive[0], &negative[0], mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			})
		}

		convey.Convey("It should match WhereGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				positive := randomShapeFloat32(length, 0x1853+int64(length))
				negative := randomShapeFloat32(length, 0x1854+int64(length))
				mask := shapeMaskBytes(length, 0x1855+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereFloat32NEONAsm(&got[0], &positive[0], &negative[0], &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			}
		})
	})
}

func TestMaskedFillF32NEONParity(t *testing.T) {
	convey.Convey("Given MaskedFillF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaskedFillGeneric for N=%d", length), func() {
				input := randomShapeFloat32(length, 0x1860+int64(length))
				mask := shapeMaskBytes(length, 0x1861+int64(length))
				fillValue := float32(-0.75)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillF32NEON(&got[0], &input[0], fillValue, mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			})
		}

		convey.Convey("It should match MaskedFillGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomShapeFloat32(length, 0x1862+int64(length))
				mask := shapeMaskBytes(length, 0x1863+int64(length))
				fillValue := float32(3.5)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillFloat32NEONAsm(&got[0], &input[0], fillValue, &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeNEONMaxULP)
			}
		})
	})
}
