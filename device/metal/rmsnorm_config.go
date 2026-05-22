package metal

import "github.com/theapemachine/manifesto/tensor"

/*
RMSNormConfig selects RMSNorm parameters for math.rmsnorm nodes.
*/
type RMSNormConfig struct {
	Epsilon float32
}

/*
DefaultRMSNormConfig matches legacy Metal RMSNorm behavior.
*/
func DefaultRMSNormConfig() RMSNormConfig {
	return RMSNormConfig{Epsilon: 1e-6}
}

/*
RunRMSNorm applies RMSNorm on Metal using config.
*/
func RunRMSNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	out tensor.Tensor,
	config RMSNormConfig,
) error {
	return runMetalRMSNormConfigured(input, scale, out, config)
}
