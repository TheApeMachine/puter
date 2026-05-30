//go:build arm64 || amd64

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
