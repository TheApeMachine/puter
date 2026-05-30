//go:build !arm64

package geometry

func dialRotate128FromTrig(out, in PhaseDial, cosine, sine float64) {
	if len(out) != len(in) || len(out) == 0 {
		return
	}

	factor := complex(cosine, sine)

	for dimIndex := range out {
		out[dimIndex] = in[dimIndex] * factor
	}
}
