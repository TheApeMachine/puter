//go:build amd64

package quant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2QuantAvailable() bool {
	return cpu.X86.HasAVX2
}

func sse2QuantAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestQuantInt8AVX2Parity(t *testing.T) {
	if !avx2QuantAvailable() {
		t.Skip("AVX2 required")
	}

	convey.Convey("Given QuantInt8AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match quantInt8Generic for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x55ac + int64(length)))
				source := make([]float32, length)

				for index := range source {
					source[index] = float32(rng.NormFloat64()) * 10
				}

				const scale = float32(0.125)
				const zeroPoint = int8(7)

				want := make([]int8, length)
				got := make([]int8, length)

				quantInt8Generic(want, source, scale, zeroPoint)
				quantInt8AVX2(got, source, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%d got=%d src=%g",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			})
		}

		convey.Convey("It should match quantInt8Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				rng := rand.New(rand.NewSource(0x55ad + int64(length)))
				source := make([]float32, length)

				for index := range source {
					source[index] = float32(rng.NormFloat64()) * 10
				}

				const scale = float32(0.125)
				const zeroPoint = int8(7)

				want := make([]int8, length)
				got := make([]int8, length)

				quantInt8Generic(want, source, scale, zeroPoint)
				QuantInt8AVX2Asm(
					&got[0], &source[0], length,
					1.0/scale, int32(zeroPoint),
				)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%d got=%d src=%g",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			}
		})
	})
}

func TestQuantInt8SSE2Parity(t *testing.T) {
	if !sse2QuantAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given QuantInt8SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match quantInt8Generic for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x55ae + int64(length)))
				source := make([]float32, length)

				for index := range source {
					source[index] = float32(rng.NormFloat64()) * 10
				}

				const scale = float32(0.125)
				const zeroPoint = int8(7)

				want := make([]int8, length)
				got := make([]int8, length)

				quantInt8Generic(want, source, scale, zeroPoint)
				quantInt8SSE2(got, source, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%d got=%d src=%g",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			})
		}

		convey.Convey("It should match quantInt8Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				rng := rand.New(rand.NewSource(0x55af + int64(length)))
				source := make([]float32, length)

				for index := range source {
					source[index] = float32(rng.NormFloat64()) * 10
				}

				const scale = float32(0.125)
				const zeroPoint = int8(7)

				want := make([]int8, length)
				got := make([]int8, length)

				quantInt8Generic(want, source, scale, zeroPoint)
				QuantInt8SSE2Asm(
					&got[0], &source[0], length,
					1.0/scale, int32(zeroPoint),
				)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%d got=%d src=%g",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			}
		})
	})
}
