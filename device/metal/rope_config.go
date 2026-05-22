package metal

import "github.com/theapemachine/manifesto/tensor"

/*
RoPEConfig selects rotary embedding parameters for positional.rope nodes.
Standard models use Base only; Llama 3 sets Type to llama3 with scaling fields.
*/
type RoPEConfig struct {
	Base            float32
	Type            string
	Mode            string
	PositionOffset  uint32
	Factor          float32
	LowFreqFactor   float32
	HighFreqFactor  float32
	OriginalContext uint32
}

/*
DefaultRoPEConfig matches legacy Metal RoPE behavior (theta=10000, no scaling).
*/
func DefaultRoPEConfig() RoPEConfig {
	return RoPEConfig{Base: 10000}
}

/*
RunRoPE applies rotary position embeddings on Metal using config.
*/
func RunRoPE(input tensor.Tensor, out tensor.Tensor, config RoPEConfig) error {
	return runMetalRoPEConfigured(input, out, config)
}

/*
RunRoPEWithPosition applies RoPE using a manifest-provided int32 position offset.
*/
func RunRoPEWithPosition(
	input tensor.Tensor,
	position tensor.Tensor,
	out tensor.Tensor,
	config RoPEConfig,
) error {
	return runMetalRoPEWithPosition(input, position, out, config)
}
