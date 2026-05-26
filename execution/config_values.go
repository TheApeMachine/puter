package execution

import (
	"strconv"

	"github.com/theapemachine/manifesto/ast"
)

func configInt(node *ast.GraphNode, key string, defaultValue int) int {
	if node == nil || node.Attributes == nil {
		return defaultValue
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return defaultValue
	}

	switch typed := raw.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case uint:
		return int(typed)
	case uint32:
		return int(typed)
	case uint64:
		return int(typed)
	case float32:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		value, err := strconv.Atoi(typed)

		if err == nil {
			return value
		}
	}

	return defaultValue
}

func configFloat(node *ast.GraphNode, key string, defaultValue float64) float64 {
	if node == nil || node.Attributes == nil {
		return defaultValue
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return defaultValue
	}

	switch typed := raw.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case string:
		value, err := strconv.ParseFloat(typed, 64)

		if err == nil {
			return value
		}
	}

	return defaultValue
}

func configBool(node *ast.GraphNode, key string, defaultValue bool) bool {
	if node == nil || node.Attributes == nil {
		return defaultValue
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return defaultValue
	}

	switch typed := raw.(type) {
	case bool:
		return typed
	case string:
		value, err := strconv.ParseBool(typed)

		if err == nil {
			return value
		}
	}

	return defaultValue
}
