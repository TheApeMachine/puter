//go:build arm64

package active_inference

import "testing"

func TestFreeEnergyDebugNEON(t *testing.T) {
	length := 4
	likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA340+int64(length))

	want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], length)
	asmGot := FreeEnergyFloat32NEONAsm(&likelihood[0], &posterior[0], &prior[0], length)
	scalarGot := FreeEnergyFloat32Scalar(likelihood, posterior, prior)
	wrapperGot := FreeEnergyF32NEON(&likelihood[0], &posterior[0], &prior[0], 7)

	t.Logf("N=4 generic=%g asm=%g scalar=%g wrapper7=%g", want, asmGot, scalarGot, wrapperGot)
}
