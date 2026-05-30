//go:build arm64

package geometry

import "math"

//go:noescape
func DialNormalize128NEONAsm(dial *complex128)

//go:noescape
func DialSimilarity128NEONAsm(left, right *complex128) float64

//go:noescape
func DialRotate128NEONAsm(out, in *complex128, cosine, sine float64)

//go:noescape
func DialAddPhases128NEONAsm(dial *complex128, cosines, sines *float64)

//go:noescape
func DialComposeMidpoint128NEONAsm(out, left, right *complex128)

func dialNormalize128Native(dial PhaseDial) {
	if len(dial) != PhaseDialDimensions {
		return
	}

	DialNormalize128NEONAsm(&dial[0])
}

func dialSimilarity128Native(left, right PhaseDial) float64 {
	if len(left) != PhaseDialDimensions || len(right) != PhaseDialDimensions {
		return 0
	}

	return DialSimilarity128NEONAsm(&left[0], &right[0])
}

func dialRotate128Native(out, in PhaseDial, angleRadians float64) {
	if len(out) != PhaseDialDimensions || len(in) != PhaseDialDimensions {
		return
	}

	cosine, sine := math.Sincos(angleRadians)
	dialRotate128FromTrigNative(out, in, cosine, sine)
}

func dialRotate128FromTrigNative(out, in PhaseDial, cosine, sine float64) {
	if len(out) != PhaseDialDimensions || len(in) != PhaseDialDimensions {
		return
	}

	dialRotate128FromTrig(out, in, cosine, sine)
}

func dialAddPhases128Native(dial PhaseDial, cosines, sines []float64) {
	if len(dial) != PhaseDialDimensions {
		return
	}

	DialAddPhases128NEONAsm(&dial[0], &cosines[0], &sines[0])
}

func dialComposeMidpoint128Native(left, right PhaseDial) PhaseDial {
	if len(left) != PhaseDialDimensions || len(right) != PhaseDialDimensions {
		return nil
	}

	out := make(PhaseDial, PhaseDialDimensions)
	DialComposeMidpoint128NEONAsm(&out[0], &left[0], &right[0])

	return out
}
