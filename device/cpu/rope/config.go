package rope

import "github.com/theapemachine/puter/device"

type RoPEConfig = device.RoPEConfig

func DefaultRoPEConfig() RoPEConfig {
	return RoPEConfig{
		BaseFreq:      10000.0,
		StartPosition: 0,
		Mode:          device.RoPEModeInterleaved,
		Scaling:       device.RoPEScalingNone,
		ScalingFactor: 1.0,
	}
}
