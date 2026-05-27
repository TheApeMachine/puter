package execution

import (
	"fmt"
	"strings"
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

func castFourPointers(values []any, method string) (
	unsafe.Pointer,
	unsafe.Pointer,
	unsafe.Pointer,
	unsafe.Pointer,
	error,
) {
	first, second, third, err := castThreePointers(values[:3], method)

	if err != nil {
		return nil, nil, nil, nil, err
	}

	fourth, err := castPointer(values[3], method, "arg3")

	if err != nil {
		return nil, nil, nil, nil, err
	}

	return first, second, third, fourth, nil
}

func castTwoInts(values []any, method string) (int, int, error) {
	first, err := castInt(values[0], method, "arg0")

	if err != nil {
		return 0, 0, err
	}

	second, err := castInt(values[1], method, "arg1")

	if err != nil {
		return 0, 0, err
	}

	return first, second, nil
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

func castFourInts(values []any, method string) (int, int, int, int, error) {
	first, second, third, err := castThreeInts(values[:3], method)

	if err != nil {
		return 0, 0, 0, 0, err
	}

	fourth, err := castInt(values[3], method, "arg3")

	if err != nil {
		return 0, 0, 0, 0, err
	}

	return first, second, third, fourth, nil
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

func castRMSNormConfig(fields map[string]any) (device.RMSNormConfig, error) {
	epsilon, err := castFloat64Field(fields, "Epsilon")

	if err != nil {
		return device.RMSNormConfig{}, err
	}

	config := device.RMSNormConfig{Epsilon: epsilon}

	if err := config.Validate(); err != nil {
		return device.RMSNormConfig{}, err
	}

	return config, nil
}

func castTimestepEmbeddingConfig(fields map[string]any) (device.TimestepEmbeddingConfig, error) {
	maxPeriod, err := castFloat64Field(fields, "MaxPeriod")

	if err != nil {
		return device.TimestepEmbeddingConfig{}, err
	}

	downscaleFreqShift, err := castFloat64Field(fields, "DownscaleFreqShift")

	if err != nil {
		return device.TimestepEmbeddingConfig{}, err
	}

	timestepDivisor, err := castFloat64Field(fields, "TimestepDivisor")

	if err != nil {
		return device.TimestepEmbeddingConfig{}, err
	}

	flipSinToCos, err := castBoolField(fields, "FlipSinToCos")

	if err != nil {
		return device.TimestepEmbeddingConfig{}, err
	}

	config := device.TimestepEmbeddingConfig{
		MaxPeriod:          float32(maxPeriod),
		DownscaleFreqShift: float32(downscaleFreqShift),
		TimestepDivisor:    float32(timestepDivisor),
		FlipSinToCos:       flipSinToCos,
	}

	if err := config.Validate(); err != nil {
		return device.TimestepEmbeddingConfig{}, err
	}

	return config, nil
}

func castConv2DConfig(fields map[string]any) (device.Conv2DConfig, error) {
	strideH, err := castIntField(fields, "StrideH")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	strideW, err := castIntField(fields, "StrideW")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	paddingH, err := castIntField(fields, "PaddingH")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	paddingW, err := castIntField(fields, "PaddingW")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	dilationH, err := castIntField(fields, "DilationH")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	dilationW, err := castIntField(fields, "DilationW")

	if err != nil {
		return device.Conv2DConfig{}, err
	}

	return device.Conv2DConfig{
		StrideH:   strideH,
		StrideW:   strideW,
		PaddingH:  paddingH,
		PaddingW:  paddingW,
		DilationH: dilationH,
		DilationW: dilationW,
	}, nil
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

	mode, err := castRoPEModeField(fields, "Mode")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	scaling, err := castRoPEScalingField(fields, "Scaling")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	scalingFactor, err := castFloat64Field(fields, "ScalingFactor")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	lowFreqFactor, err := castFloat64Field(fields, "LowFreqFactor")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	highFreqFactor, err := castFloat64Field(fields, "HighFreqFactor")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	originalContext, err := castIntField(fields, "OriginalContext")

	if err != nil {
		return device.RoPEConfig{}, err
	}

	return device.RoPEConfig{
		BaseFreq:        baseFreq,
		StartPosition:   startPosition,
		Mode:            mode,
		Scaling:         scaling,
		ScalingFactor:   scalingFactor,
		LowFreqFactor:   lowFreqFactor,
		HighFreqFactor:  highFreqFactor,
		OriginalContext: originalContext,
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

func castStringField(fields map[string]any, name string) (string, error) {
	value, ok := fields[name]

	if !ok {
		return "", fmt.Errorf("router config: missing %q", name)
	}

	asString, ok := value.(string)

	if !ok {
		return "", fmt.Errorf("router config %q is %T, expected string", name, value)
	}

	return asString, nil
}

func castRoPEModeField(fields map[string]any, name string) (device.RoPEMode, error) {
	value, err := castStringField(fields, name)

	if err != nil {
		return device.RoPEModeInterleaved, err
	}

	switch strings.ToLower(value) {
	case "interleaved":
		return device.RoPEModeInterleaved, nil
	case "half":
		return device.RoPEModeHalf, nil
	default:
		return device.RoPEModeInterleaved, fmt.Errorf("router config %q has unsupported RoPE mode %q", name, value)
	}
}

func castRoPEScalingField(fields map[string]any, name string) (device.RoPEScaling, error) {
	value, err := castStringField(fields, name)

	if err != nil {
		return device.RoPEScalingNone, err
	}

	switch strings.ToLower(value) {
	case "none":
		return device.RoPEScalingNone, nil
	case "llama3":
		return device.RoPEScalingLlama3, nil
	default:
		return device.RoPEScalingNone, fmt.Errorf("router config %q has unsupported RoPE scaling %q", name, value)
	}
}
