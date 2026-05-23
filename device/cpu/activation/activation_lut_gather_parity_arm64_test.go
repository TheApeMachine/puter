//go:build arm64

package activation

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestActivationLUTGatherNEONParity(t *testing.T) {
	convey.Convey("Given f16/bf16 LUT NEON gather kernels", t, func() {
		lutTables := []struct {
			name string
			lut  *[65536]uint16
		}{
			{"Exp", &expF16LUT},
			{"ReLU", &reluF16LUT},
			{"Tanh", &tanhF16LUT},
			{"Sigmoid", &sigmoidF16LUT},
			{"Gelu", &geluF16LUT},
		}

		for _, table := range lutTables {
			convey.Convey(table.name, func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						source := make([]uint16, count)
						want := make([]uint16, count)
						got := make([]uint16, count)

						rng := rand.New(rand.NewSource(int64(0x7100 + count)))
						for index := range source {
							source[index] = uint16(rng.Intn(65536))
						}

						applyF16LUTScalar(&want[0], &source[0], count, table.lut)
						f16LUTGatherKernel(&got[0], &source[0], count, table.lut)

						for index := range want {
							if got[index] != want[index] {
								t.Fatalf(
									"LUT gather mismatch at %d: got=%04x want=%04x",
									index, got[index], want[index],
								)
							}
						}
					})
				}
			})
		}
	})
}

func BenchmarkActivationLUTGatherNEONExp(b *testing.B) {
	count := 8192
	source := make([]uint16, count)
	destination := make([]uint16, count)
	rng := rand.New(rand.NewSource(1))

	for index := range source {
		source[index] = uint16(rng.Intn(65536))
	}

	b.ResetTimer()

	for b.Loop() {
		f16LUTGatherKernel(&destination[0], &source[0], count, &expF16LUT)
	}
}
