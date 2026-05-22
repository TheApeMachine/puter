package optimizer

/*
f32Mul and f32Add prevent the arm64 compiler from fusing Adam moment
updates into FMA instructions that differ from the explicit mul/add
sequence used in the NEON kernels.
*/
//go:noinline
func f32Mul(left, right float32) float32 {
	return left * right
}

//go:noinline
func f32Add(left, right float32) float32 {
	return left + right
}

//go:noinline
func adamFirstMomentUpdate(beta1, first, grad float32) float32 {
	oneMinusBeta1 := f32Add(1, -beta1)

	return f32Add(f32Mul(beta1, first), f32Mul(oneMinusBeta1, grad))
}

//go:noinline
func adamSecondMomentUpdate(beta2, second, gradSquared float32) float32 {
	oneMinusBeta2 := f32Add(1, -beta2)

	return f32Add(f32Mul(beta2, second), f32Mul(oneMinusBeta2, gradSquared))
}
