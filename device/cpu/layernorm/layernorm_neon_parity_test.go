//go:build arm64

package layernorm

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const layerNormNEONMaxULP = 1

func randomLayerNormRow(length int, seed int64) ([]float32, []float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	row := make([]float32, length)
	scale := make([]float32, length)
	bias := make([]float32, length)

	for index := range row {
		row[index] = float32((rng.Float64() - 0.5) * 4)
		scale[index] = float32(rng.Float64()*0.5 + 0.75)
		bias[index] = float32((rng.Float64() - 0.5) * 0.25)
	}

	return row, scale, bias
}

func TestLayerNormApplyRowF32NEONParity(t *testing.T) {
	convey.Convey("Given LayerNormApplyRow float32 NEON", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, scale, bias := randomLayerNormRow(length, 0x4E10+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				LayerNormApplyRowNative(got, row, scale, bias, mean, invStdDev)

				parity.AssertFloat32SlicesWithinULP(t, got, want, layerNormNEONMaxULP)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				if length < 4 {
					continue
				}

				row, scale, bias := randomLayerNormRow(length, 0x4E11+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				blockCount := length &^ 3
				LayerNormApplyRowNEONAsm(
					&got[0], &row[0], &scale[0], &bias[0],
					blockCount, mean, invStdDev,
				)
				for index := blockCount; index < length; index++ {
					delta := row[index] - mean
					delta *= invStdDev
					got[index] = scale[index]*delta + bias[index]
				}

				parity.AssertFloat32SlicesWithinULP(t, got, want, layerNormNEONMaxULP)
			}
		})
	})
}
