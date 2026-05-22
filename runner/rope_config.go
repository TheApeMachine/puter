package runner

import (
	"strconv"

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

func nodeOptionalFloatAttribute(node *ir.Node, key string) (float64, bool) {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeFloat {
		parsed, err := strconv.ParseFloat(attribute.Value, 64)

		if err == nil {
			return parsed, true
		}
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			switch typed := raw.(type) {
			case float64:
				return typed, true
			case float32:
				return float64(typed), true
			case int:
				return float64(typed), true
			case int64:
				return float64(typed), true
			}
		}
	}

	return 0, false
}

func nodeOptionalStringAttribute(node *ir.Node, key string) (string, bool) {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeString {
		return attribute.Value, true
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			if typed, ok := raw.(string); ok {
				return typed, true
			}
		}
	}

	return "", false
}

func nodeOptionalIntAttribute(node *ir.Node, key string) (int, bool) {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeInt {
		parsed, err := strconv.ParseInt(attribute.Value, 10, 64)

		if err == nil {
			return int(parsed), true
		}
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			switch typed := raw.(type) {
			case int:
				return typed, true
			case int64:
				return int(typed), true
			case float64:
				return int(typed), true
			}
		}
	}

	return 0, false
}
