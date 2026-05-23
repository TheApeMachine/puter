package math

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestFastExp64Log2ERegression(t *testing.T) {
	convey.Convey("Given the Metal parity regression input", t, func() {
		input := 1.9228818
		got := dtype.Fromfloat32(float32(FastExp64(input)))
		want := dtype.Fromfloat32(float32(stdmath.Exp(input)))

		if got.Bits() != want.Bits() {
			t.Fatalf("f16 bits got=%04x want=%04x", got.Bits(), want.Bits())
		}
	})
}

func TestFastExp64Determinism(t *testing.T) {
	convey.Convey("Given FastExp64", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := randomMathFloat32(count, 0x2310+int64(count))

				for index, input := range source {
					first := FastExp64(float64(input))
					second := FastExp64(float64(input))

					if first != second {
						t.Fatalf("lane %d non-deterministic got=%g second=%g", index, first, second)
					}
				}
			})
		}
	})
}

func TestFastSinh64PathParity(t *testing.T) {
	convey.Convey("Given FastSinh64 built from FastExp64", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := randomMathFloat32(count, 0x2311+int64(count))

				for index, input := range source {
					value := float64(input)
					got := FastSinh64(value)
					want := 0.5 * (FastExp64(value) - FastExp64(-value))

					if got != want {
						t.Fatalf(
							"lane %d sinh path mismatch got=%g want=%g input=%g",
							index, got, want, input,
						)
					}
				}
			})
		}
	})
}

func BenchmarkFastExp64(b *testing.B) {
	value := 1.9228818

	for b.Loop() {
		_ = FastExp64(value)
	}
}

func BenchmarkFastSinh64(b *testing.B) {
	value := 1.9228818

	for b.Loop() {
		_ = FastSinh64(value)
	}
}
