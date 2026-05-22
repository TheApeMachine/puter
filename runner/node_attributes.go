package runner

import (
	"strconv"

	"github.com/theapemachine/manifesto/ir"
)

func nodeOptionalFloatAttribute(node *ir.Node, key string) (float64, bool) {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeFloat {
		parsed, err := strconv.ParseFloat(attribute.Value, 64)

		if err == nil {
			return parsed, true
		}
	}

	if attribute.Kind == ir.AttributeInt {
		parsed, err := strconv.ParseInt(attribute.Value, 10, 64)

		if err == nil {
			return float64(parsed), true
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

func rmsNormEpsilonFromNode(node *ir.Node) float32 {
	if eps, ok := nodeOptionalFloatAttribute(node, "eps"); ok && eps > 0 {
		return float32(eps)
	}

	return 1e-6
}
