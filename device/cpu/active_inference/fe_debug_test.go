//go:build arm64

package active_inference

import (
	"fmt"
	"testing"
)

func TestFEDebugSeeds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		seed   int64
		length int
		block  int
	}{
		{"parity7", 0xA307, 7, 4},
		{"block4", 0xA344, 4, 4},
		{"uniform", 0, 4, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "uniform" {
				like := []float32{0.25, 0.25, 0.25, 0.25}
				post := []float32{0.25, 0.25, 0.25, 0.25}
				prior := []float32{0.25, 0.25, 0.25, 0.25}
				want := FreeEnergyF32Generic(&like[0], &post[0], &prior[0], tc.block)
				got := FreeEnergyFloat32NEONAsm(&like[0], &post[0], &prior[0], tc.block)
				fmt.Printf("%s asm=%g want=%g neon_wrap=%g\n", tc.name, got, want, FreeEnergyF32NEON(&like[0], &post[0], &prior[0], tc.length))
				return
			}

			likelihood, posterior, prior := randomActiveInferenceVectors(tc.length, tc.seed)
			want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], tc.block)
			got := FreeEnergyFloat32NEONAsm(&likelihood[0], &posterior[0], &prior[0], tc.block)
			gotAgain := FreeEnergyFloat32NEONAsm(&likelihood[0], &posterior[0], &prior[0], tc.block)
			wrap := FreeEnergyF32NEON(&likelihood[0], &posterior[0], &prior[0], tc.length)
			if got != gotAgain {
				t.Fatalf("%s asm non-deterministic: %g vs %g", tc.name, got, gotAgain)
			}
			fmt.Printf("%s asm=%g again=%g want=%g wrap=%g\n", tc.name, got, gotAgain, want, wrap)
		})
	}
}
