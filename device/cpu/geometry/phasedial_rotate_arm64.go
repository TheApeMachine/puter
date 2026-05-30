//go:build arm64

package geometry

func dialRotate128FromTrig(out, in PhaseDial, cosine, sine float64) {
	DialRotate128NEONAsm(&out[0], &in[0], cosine, sine)
}
