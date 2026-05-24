package math

import "math"

//go:noinline
func fastExp32Eval(x float32) float32 {
	z := x * float32(1.4426950408889634)

	k := int32(z)

	if z < 0 {
		k--
	}

	f := z - float32(k)

	poly := float32(1.0) + f*(float32(0.69314718)+f*(float32(0.24022650)+f*(float32(0.05550410)+f*(float32(0.00961812)+f*float32(0.00133389)))))

	bits := math.Float32bits(poly)
	bits += uint32(k) << 23

	return math.Float32frombits(bits)
}

// FastExp32 computes e^x using IEEE-754 bit manipulation and a polynomial approximation.
// It avoids float64 and is roughly 5-10x faster than math.Exp in Go.
func FastExp32(x float32) float32 {
	if x < -87.33654 {
		return 0.0
	}

	if x > 88.72283 {
		return 0x1p127 * (1 + (1 - 0x1p-23))
	}

	return fastExp32Eval(x)
}
