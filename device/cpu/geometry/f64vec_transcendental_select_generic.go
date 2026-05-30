//go:build !amd64 && !arm64

package geometry

var transcendentalFloat64Funcs = []f64TranscendentalKernelImpl{
	{
		sinCos:     vecSinCosFloat64Scalar,
		cosine:     vecCosFloat64Scalar,
		arcTangent: vecAtan2Float64Scalar,
		name:       "generic",
		available:  true,
	},
}
