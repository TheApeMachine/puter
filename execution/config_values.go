package execution

import (
	"fmt"
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

func configInts(node *ast.GraphNode, key string, defaultValue []int) ([]int, error) {
	if node == nil || node.Attributes == nil {
		return defaultValue, nil
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return defaultValue, nil
	}

	switch typed := raw.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		return intSliceValues(typed)
	default:
		return nil, fmt.Errorf("config %q is %T, expected int[]", key, raw)
	}
}

func intSliceValues(raw []any) ([]int, error) {
	values := make([]int, len(raw))

	for index, value := range raw {
		converted, err := configScalarInt(value)

		if err != nil {
			return nil, fmt.Errorf("config int[%d]: %w", index, err)
		}

		values[index] = converted
	}

	return values, nil
}

func configScalarInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case uint:
		return int(typed), nil
	case uint32:
		return int(typed), nil
	case uint64:
		return int(typed), nil
	case float32:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		value, err := strconv.Atoi(typed)

		if err != nil {
			return 0, err
		}

		return value, nil
	default:
		return 0, fmt.Errorf("%T is not supported", value)
	}
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

func configString(node *ast.GraphNode, key string, defaultValue string) string {
	if node == nil || node.Attributes == nil {
		return defaultValue
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return defaultValue
	}

	asString, ok := raw.(string)

	if !ok {
		return defaultValue
	}

	return asString
}
