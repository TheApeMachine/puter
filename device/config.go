package device

import "fmt"

/*
Shared configuration structs referenced by device interfaces. Domain
packages define matching types locally; the field layout must stay in
sync with these definitions.
*/

type DropoutConfig struct {
	Rate float32
	Seed uint64
}

type PoolConfig struct {
	KernelH  int
	KernelW  int
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
}

type Conv2DConfig struct {
	StrideH   int
	StrideW   int
	PaddingH  int
	PaddingW  int
	DilationH int
	DilationW int
}

type Conv1DConfig struct {
	Stride   int
	Padding  int
	Dilation int
}

type Conv3DConfig struct {
	StrideD, StrideH, StrideW       int
	PaddingD, PaddingH, PaddingW    int
	DilationD, DilationH, DilationW int
}

type SamplingConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        uint64
}

type GroupNormConfig struct {
	Groups int
}

type RMSNormConfig struct {
	Epsilon float64
}

func (config RMSNormConfig) Validate() error {
	if !(config.Epsilon > 0) {
		return fmt.Errorf("rmsnorm: epsilon must be positive, got %g", config.Epsilon)
	}

	return nil
}

type RoPEMode int

const (
	RoPEModeInterleaved RoPEMode = iota
	RoPEModeHalf
)

type RoPEScaling int

const (
	RoPEScalingNone RoPEScaling = iota
	RoPEScalingLlama3
)

type RoPEConfig struct {
	BaseFreq        float64
	StartPosition   int
	Mode            RoPEMode
	Scaling         RoPEScaling
	ScalingFactor   float64
	LowFreqFactor   float64
	HighFreqFactor  float64
	OriginalContext int
}

func (config RoPEConfig) Validate() error {
	if config.BaseFreq <= 0 {
		return fmt.Errorf("rope: base frequency must be positive, got %g", config.BaseFreq)
	}

	switch config.Mode {
	case RoPEModeInterleaved, RoPEModeHalf:
	default:
		return fmt.Errorf("rope: unsupported mode %d", config.Mode)
	}

	switch config.Scaling {
	case RoPEScalingNone:
		return nil
	case RoPEScalingLlama3:
		return config.validateLlama3Scaling()
	default:
		return fmt.Errorf("rope: unsupported scaling %d", config.Scaling)
	}
}

func (config RoPEConfig) validateLlama3Scaling() error {
	if config.ScalingFactor <= 0 {
		return fmt.Errorf("rope: scaling factor must be positive, got %g", config.ScalingFactor)
	}

	if config.LowFreqFactor <= 0 {
		return fmt.Errorf("rope: low frequency factor must be positive, got %g", config.LowFreqFactor)
	}

	if config.HighFreqFactor <= config.LowFreqFactor {
		return fmt.Errorf(
			"rope: high frequency factor %g must exceed low frequency factor %g",
			config.HighFreqFactor,
			config.LowFreqFactor,
		)
	}

	if config.OriginalContext <= 0 {
		return fmt.Errorf("rope: original context must be positive, got %d", config.OriginalContext)
	}

	return nil
}

type VSAConfig struct {
	Shift int
}

type PredictiveCodingConfig struct {
	LearningRate float32
}

type DequantInt8Config struct {
	Scale     float32
	ZeroPoint int8
}

type DequantInt4Config struct {
	Scale     float32
	ZeroPoint int8
}

type FlashAttentionConfig struct {
	BlockSize int
	Causal    bool
}

type MultiHeadAttentionConfig struct {
	NumHeads    int
	HeadDim     int
	Causal      bool
	WindowSize  int
	ALiBiSlope  float32
	KVHeadCount int
}
