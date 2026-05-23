//go:build arm64

package active_inference

//go:noescape
func activeInferenceLogF64NEONAsm(value float64) float64
