package sampling

import "github.com/theapemachine/puter/device"

type SamplingConfig = device.SamplingConfig

func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{Temperature: 1.0, TopK: 0, TopP: 1.0, Seed: 0xfeedface}
}
