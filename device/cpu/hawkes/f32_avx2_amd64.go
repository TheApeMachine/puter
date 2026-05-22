//go:build amd64

package hawkes

//go:noescape
func HawkesExpSumFloat32AVX2Asm(exponents *float32, count int) float32

//go:noescape
func HawkesScaledExpStoreFloat32AVX2Asm(
	exponents *float32,
	alpha float32,
	out *float32,
	count int,
)
