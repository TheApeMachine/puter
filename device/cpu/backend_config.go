package cpu

import (
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/cpu/convolution"
	"github.com/theapemachine/puter/device/cpu/dropout"
	"github.com/theapemachine/puter/device/cpu/normalization"
	"github.com/theapemachine/puter/device/cpu/pool"
	"github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/cpu/rope"
	"github.com/theapemachine/puter/device/cpu/sampling"
	"github.com/theapemachine/puter/device/cpu/vsa"
)

func poolConfig(config device.PoolConfig) pool.PoolConfig {
	return pool.PoolConfig{
		KernelH:  config.KernelH,
		KernelW:  config.KernelW,
		StrideH:  config.StrideH,
		StrideW:  config.StrideW,
		PaddingH: config.PaddingH,
		PaddingW: config.PaddingW,
	}
}

func conv2DConfig(config device.Conv2DConfig) convolution.Conv2DConfig {
	return convolution.Conv2DConfig{
		StrideH:   config.StrideH,
		StrideW:   config.StrideW,
		PaddingH:  config.PaddingH,
		PaddingW:  config.PaddingW,
		DilationH: config.DilationH,
		DilationW: config.DilationW,
	}
}

func conv1DConfig(config device.Conv1DConfig) convolution.Conv1DConfig {
	return convolution.Conv1DConfig{
		Stride:   config.Stride,
		Padding:  config.Padding,
		Dilation: config.Dilation,
	}
}

func conv3DConfig(config device.Conv3DConfig) convolution.Conv3DConfig {
	return convolution.Conv3DConfig{
		StrideD:    config.StrideD,
		StrideH:    config.StrideH,
		StrideW:    config.StrideW,
		PaddingD:   config.PaddingD,
		PaddingH:   config.PaddingH,
		PaddingW:   config.PaddingW,
		DilationD:  config.DilationD,
		DilationH:  config.DilationH,
		DilationW:  config.DilationW,
	}
}

func dropoutConfig(config device.DropoutConfig) dropout.DropoutConfig {
	return dropout.DropoutConfig{Rate: config.Rate, Seed: config.Seed}
}

func samplingConfig(config device.SamplingConfig) sampling.SamplingConfig {
	return sampling.SamplingConfig{
		Temperature: config.Temperature,
		TopK:        config.TopK,
		TopP:        config.TopP,
		Seed:        config.Seed,
	}
}

func groupNormConfig(config device.GroupNormConfig) normalization.GroupNormConfig {
	return normalization.GroupNormConfig{Groups: config.Groups}
}

func ropeConfig(config device.RoPEConfig) rope.RoPEConfig {
	return rope.RoPEConfig{
		BaseFreq:      config.BaseFreq,
		StartPosition: config.StartPosition,
	}
}

func flashAttentionConfig(config device.FlashAttentionConfig) attention.FlashAttentionConfig {
	return attention.FlashAttentionConfig{
		BlockSize: config.BlockSize,
		Causal:    config.Causal,
	}
}

func multiHeadAttentionConfig(config device.MultiHeadAttentionConfig) attention.MultiHeadAttentionConfig {
	return attention.MultiHeadAttentionConfig{
		NumHeads:    config.NumHeads,
		HeadDim:     config.HeadDim,
		Causal:      config.Causal,
		WindowSize:  config.WindowSize,
		ALiBiSlope:  config.ALiBiSlope,
		KVHeadCount: config.KVHeadCount,
	}
}

func vsaConfig(config device.VSAConfig) vsa.VSAConfig {
	return vsa.VSAConfig{Shift: config.Shift}
}

func predictiveCodingConfig(config device.PredictiveCodingConfig) predictive_coding.PredictiveCodingConfig {
	return predictive_coding.PredictiveCodingConfig{LearningRate: config.LearningRate}
}
