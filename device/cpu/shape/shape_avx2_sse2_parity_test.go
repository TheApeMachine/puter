//go:build amd64

package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const shapeAVX2SSE2MaxULP = 0

func avx2ShapeAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2ShapeAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestCopyContiguousF32AVX2Parity(t *testing.T) {
	if !avx2ShapeAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given CopyContiguousF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CopyContiguousGeneric for N=%d", length), func() {
				source := randomShapeFloat32(length, 0x1840+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousF32AVX2(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match CopyContiguousGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomShapeFloat32(length, 0x1841+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousFloat32AVX2Asm(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}

func TestCopyContiguousF32SSE2Parity(t *testing.T) {
	if !sse2ShapeAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given CopyContiguousF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CopyContiguousGeneric for N=%d", length), func() {
				source := randomShapeFloat32(length, 0x1850+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousF32SSE2(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match CopyContiguousGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomShapeFloat32(length, 0x1851+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousFloat32SSE2Asm(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}

func TestWhereF32AVX2Parity(t *testing.T) {
	if !avx2ShapeAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given WhereF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WhereGeneric for N=%d", length), func() {
				positive := randomShapeFloat32(length, 0x1860+int64(length))
				negative := randomShapeFloat32(length, 0x1861+int64(length))
				mask := shapeMaskBytes(length, 0x1862+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereF32AVX2(&got[0], &positive[0], &negative[0], mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match WhereGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				positive := randomShapeFloat32(length, 0x1863+int64(length))
				negative := randomShapeFloat32(length, 0x1864+int64(length))
				mask := shapeMaskBytes(length, 0x1865+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereFloat32AVX2Asm(&got[0], &positive[0], &negative[0], &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}

func TestWhereF32SSE2Parity(t *testing.T) {
	if !sse2ShapeAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given WhereF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WhereGeneric for N=%d", length), func() {
				positive := randomShapeFloat32(length, 0x1870+int64(length))
				negative := randomShapeFloat32(length, 0x1871+int64(length))
				mask := shapeMaskBytes(length, 0x1872+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereF32SSE2(&got[0], &positive[0], &negative[0], mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match WhereGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				positive := randomShapeFloat32(length, 0x1873+int64(length))
				negative := randomShapeFloat32(length, 0x1874+int64(length))
				mask := shapeMaskBytes(length, 0x1875+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereFloat32SSE2Asm(&got[0], &positive[0], &negative[0], &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}

func TestMaskedFillF32AVX2Parity(t *testing.T) {
	if !avx2ShapeAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given MaskedFillF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaskedFillGeneric for N=%d", length), func() {
				input := randomShapeFloat32(length, 0x1880+int64(length))
				mask := shapeMaskBytes(length, 0x1881+int64(length))
				fillValue := float32(-0.75)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillF32AVX2(&got[0], &input[0], fillValue, mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match MaskedFillGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomShapeFloat32(length, 0x1882+int64(length))
				mask := shapeMaskBytes(length, 0x1883+int64(length))
				fillValue := float32(3.5)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillFloat32AVX2Asm(&got[0], &input[0], fillValue, &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}

func TestMaskedFillF32SSE2Parity(t *testing.T) {
	if !sse2ShapeAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given MaskedFillF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaskedFillGeneric for N=%d", length), func() {
				input := randomShapeFloat32(length, 0x1890+int64(length))
				mask := shapeMaskBytes(length, 0x1891+int64(length))
				fillValue := float32(-0.75)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillF32SSE2(&got[0], &input[0], fillValue, mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match MaskedFillGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomShapeFloat32(length, 0x1892+int64(length))
				mask := shapeMaskBytes(length, 0x1893+int64(length))
				fillValue := float32(3.5)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillFloat32SSE2Asm(&got[0], &input[0], fillValue, &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX2SSE2MaxULP)
			}
		})
	})
}
