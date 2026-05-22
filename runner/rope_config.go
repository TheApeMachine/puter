package runner

import (
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/puter/device/metal"
)

func ropeConfigFromNode(node *ir.Node) metal.RoPEConfig {
	config := metal.DefaultRoPEConfig()

	if base, ok := nodeOptionalFloatAttribute(node, "base"); ok {
		config.Base = float32(base)
	}

	if ropeType, ok := nodeOptionalStringAttribute(node, "rope_type"); ok {
		config.Type = ropeType
	}

	if mode, ok := nodeOptionalStringAttribute(node, "mode"); ok {
		config.Mode = mode
	}

	if factor, ok := nodeOptionalFloatAttribute(node, "rope_factor"); ok {
		config.Factor = float32(factor)
	}

	if lowFreq, ok := nodeOptionalFloatAttribute(node, "rope_low_freq_factor"); ok {
		config.LowFreqFactor = float32(lowFreq)
	}

	if highFreq, ok := nodeOptionalFloatAttribute(node, "rope_high_freq_factor"); ok {
		config.HighFreqFactor = float32(highFreq)
	}

	if originalContext, ok := nodeOptionalIntAttribute(node, "rope_original_context"); ok {
		config.OriginalContext = uint32(originalContext)
	}

	return config
}
