package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func castTwoPointers(values []any, method string) (unsafe.Pointer, unsafe.Pointer, error) {
	first, err := castPointer(values[0], method, "arg0")

	if err != nil {
		return nil, nil, err
	}

	second, err := castPointer(values[1], method, "arg1")

	if err != nil {
		return nil, nil, err
	}

	return first, second, nil
}

func castThreePointers(values []any, method string) (
	unsafe.Pointer,
	unsafe.Pointer,
	unsafe.Pointer,
	error,
) {
	first, second, err := castTwoPointers(values[:2], method)

	if err != nil {
		return nil, nil, nil, err
	}

	third, err := castPointer(values[2], method, "arg2")

	if err != nil {
		return nil, nil, nil, err
	}

	return first, second, third, nil
}

func castThreeInts(values []any, method string) (int, int, int, error) {
	first, err := castInt(values[0], method, "arg0")

	if err != nil {
		return 0, 0, 0, err
	}

	second, err := castInt(values[1], method, "arg1")

	if err != nil {
		return 0, 0, 0, err
	}

	third, err := castInt(values[2], method, "arg2")

	if err != nil {
		return 0, 0, 0, err
	}

	return first, second, third, nil
}

func castPointer(value any, method, parameter string) (unsafe.Pointer, error) {
	pointer, ok := value.(unsafe.Pointer)

	if !ok {
		return nil, fmt.Errorf("router %s: arg %q is %T, expected unsafe.Pointer", method, parameter, value)
	}

	return pointer, nil
}

func castInt(value any, method, parameter string) (int, error) {
	asInt, ok := value.(int)

	if !ok {
		return 0, fmt.Errorf("router %s: arg %q is %T, expected int", method, parameter, value)
	}

	return asInt, nil
}

func castDType(value any, method, parameter string) (dtype.DType, error) {
	asDType, ok := value.(dtype.DType)

	if !ok {
		return dtype.Invalid, fmt.Errorf("router %s: arg %q is %T, expected dtype.DType", method, parameter, value)
	}

	return asDType, nil
}

func castRoPEConfig(fields map[string]any) (device.RoPEConfig, error) {
	baseFreq, err := castFloat64Field(fields, "BaseFreq")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	startPosition, err := castIntField(fields, "StartPosition")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	return device.RoPEConfig{
		BaseFreq:      baseFreq,
		StartPosition: startPosition,
	}, nil
}

func castMultiHeadAttentionConfig(fields map[string]any) (device.MultiHeadAttentionConfig, error) {
	numHeads, err := castIntField(fields, "NumHeads")

	if err != nil {
		return device.MultiHeadAttentionConfig{}, err
	}

	headDim, err := castIntField(fields, "HeadDim")

	if err != nil {
		return device.MultiHeadAttentionConfig{}, err
	}

	causal, err := castBoolField(fields, "Causal")

	if err != nil {
		return device.MultiHeadAttentionConfig{}, err
	}

	kvHeadCount, err := castIntField(fields, "KVHeadCount")

	if err != nil {
		return device.MultiHeadAttentionConfig{}, err
	}

	return device.MultiHeadAttentionConfig{
		NumHeads:    numHeads,
		HeadDim:     headDim,
		Causal:      causal,
		KVHeadCount: kvHeadCount,
	}, nil
}

func castFloat64Field(fields map[string]any, name string) (float64, error) {
	value, ok := fields[name]

	if !ok {
		return 0, fmt.Errorf("router config: missing %q", name)
	}

	switch typed := value.(type) {
	case float32:
		return float64(typed), nil
	case float64:
		return typed, nil
	default:
		return 0, fmt.Errorf("router config %q is %T, expected float", name, value)
	}
}

func castIntField(fields map[string]any, name string) (int, error) {
	value, ok := fields[name]

	if !ok {
		return 0, fmt.Errorf("router config: missing %q", name)
	}

	return castInt(value, "config", name)
}

func castBoolField(fields map[string]any, name string) (bool, error) {
	value, ok := fields[name]

	if !ok {
		return false, fmt.Errorf("router config: missing %q", name)
	}

	asBool, ok := value.(bool)

	if !ok {
		return false, fmt.Errorf("router config %q is %T, expected bool", name, value)
	}

	return asBool, nil
}
