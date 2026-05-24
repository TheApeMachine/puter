package normalization

import "github.com/theapemachine/puter/device"

type GroupNormConfig = device.GroupNormConfig

func DefaultGroupNormConfig() GroupNormConfig {
	return GroupNormConfig{Groups: 32}
}
