package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type binaryDeviceCall func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int, dtype.DType)
type unaryDeviceCall func(unsafe.Pointer, unsafe.Pointer, int, dtype.DType)

type batchNormDenormDevice interface {
	BatchNormDenorm(
		input, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
}

func callRouter(
	deviceBackend executionDevice,
	bind OperationBind,
	configFields map[string]any,
	args []any,
) error {
	switch bind.Method {
	case "Lookup":
		return callLookup(deviceBackend, args)
	case "TimestepEmbedding":
		return callTimestepEmbedding(deviceBackend, configFields, args)
	case "Matmul":
		return callMatmul(deviceBackend, args)
	case "Conv2D":
		return callConv2D(deviceBackend, configFields, args)
	case "RMSNorm":
		return callRMSNorm(deviceBackend, configFields, args)
	case "AdaptiveRMSNorm":
		return callAdaptiveRMSNorm(deviceBackend, configFields, args)
	case "LayerNorm":
		return callLayerNorm(deviceBackend, args)
	case "GroupNorm":
		return callGroupNorm(deviceBackend, configFields, args)
	case "ModulatedLayerNorm":
		return callModulatedLayerNorm(deviceBackend, configFields, args)
	case "BatchNormDenorm":
		return callBatchNormDenorm(deviceBackend, args)
	case "Add":
		return callBinary("Add", args, deviceBackend.Add)
	case "Sub":
		return callBinary("Sub", args, deviceBackend.Sub)
	case "Mul":
		return callBinary("Mul", args, deviceBackend.Mul)
	case "Div":
		return callBinary("Div", args, deviceBackend.Div)
	case "ReLU":
		return callUnary("ReLU", args, deviceBackend.ReLU)
	case "Sigmoid":
		return callUnary("Sigmoid", args, deviceBackend.Sigmoid)
	case "Tanh":
		return callUnary("Tanh", args, deviceBackend.Tanh)
	case "Gelu":
		return callUnary("Gelu", args, deviceBackend.Gelu)
	case "Silu":
		return callUnary("Silu", args, deviceBackend.Silu)
	case "SwiGLU":
		return callSwiGLU(deviceBackend, args)
	case "SwiGLUTensors":
		return callSwiGLUTensors(deviceBackend, args)
	case "RoPE":
		return callRoPE(deviceBackend, configFields, args)
	case "MultiAxisRoPE":
		return callMultiAxisRoPE(deviceBackend, configFields, args)
	case "MultiHeadAttention":
		return callMultiHeadAttention(deviceBackend, configFields, args)
	default:
		return fmt.Errorf("router: unknown method %q", bind.Method)
	}
}

func callBatchNormDenorm(deviceBackend executionDevice, args []any) error {
	if len(args) != 8 {
		return fmt.Errorf("router: BatchNormDenorm expects 8 args, got %d", len(args))
	}

	normalizationDevice, ok := deviceBackend.(batchNormDenormDevice)

	if !ok {
		return fmt.Errorf("router: backend %T does not implement BatchNormDenorm", deviceBackend)
	}

	input, mean, variance, output, err := castFourPointers(args[:4], "BatchNormDenorm")

	if err != nil {
		return err
	}

	batch, channels, spatial, err := castThreeInts(args[4:7], "BatchNormDenorm")

	if err != nil {
		return err
	}

	format, err := castDType(args[7], "BatchNormDenorm", "format")

	if err != nil {
		return err
	}

	normalizationDevice.BatchNormDenorm(input, mean, variance, output, batch, channels, spatial, format)

	return nil
}

func callLookup(deviceBackend executionDevice, args []any) error {
	if len(args) != 7 {
		return fmt.Errorf("router: Lookup expects 7 args, got %d", len(args))
	}

	table, indices, output, err := castThreePointers(args[:3], "Lookup")

	if err != nil {
		return err
	}

	vocab, hidden, indexCount, err := castThreeInts(args[3:6], "Lookup")

	if err != nil {
		return err
	}

	format, err := castDType(args[6], "Lookup", "format")

	if err != nil {
		return err
	}

	deviceBackend.Lookup(table, indices, output, vocab, hidden, indexCount, format)

	return nil
}

func callTimestepEmbedding(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 5 {
		return fmt.Errorf("router: TimestepEmbedding expects 5 args, got %d", len(args))
	}

	config, err := castTimestepEmbeddingConfig(configFields)

	if err != nil {
		return err
	}

	timesteps, output, err := castTwoPointers(args[:2], "TimestepEmbedding")

	if err != nil {
		return err
	}

	count, dim, err := castTwoInts(args[2:4], "TimestepEmbedding")

	if err != nil {
		return err
	}

	format, err := castDType(args[4], "TimestepEmbedding", "format")

	if err != nil {
		return err
	}

	deviceBackend.TimestepEmbedding(config, timesteps, output, count, dim, format)

	return nil
}

func callMatmul(deviceBackend executionDevice, args []any) error {
	if len(args) != 7 {
		return fmt.Errorf("router: Matmul expects 7 args, got %d", len(args))
	}

	output, left, right, err := castThreePointers(args[:3], "Matmul")

	if err != nil {
		return err
	}

	rows, inner, cols, err := castThreeInts(args[3:6], "Matmul")

	if err != nil {
		return err
	}

	format, err := castDType(args[6], "Matmul", "format")

	if err != nil {
		return err
	}

	deviceBackend.Matmul(output, left, right, rows, inner, cols, format)

	return nil
}

func callConv2D(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 14 {
		return fmt.Errorf("router: Conv2D expects 14 args, got %d", len(args))
	}

	config, err := castConv2DConfig(configFields)

	if err != nil {
		return err
	}

	input, weight, bias, output, err := castFourPointers(args[:4], "Conv2D")

	if err != nil {
		return err
	}

	batch, inChannels, inHeight, inWidth, err := castFourInts(args[4:8], "Conv2D")

	if err != nil {
		return err
	}

	outChannels, kernelHeight, kernelWidth, err := castThreeInts(args[8:11], "Conv2D")

	if err != nil {
		return err
	}

	outHeight, outWidth, err := castTwoInts(args[11:13], "Conv2D")

	if err != nil {
		return err
	}

	format, err := castDType(args[13], "Conv2D", "format")

	if err != nil {
		return err
	}

	deviceBackend.Conv2D(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	return nil
}

func callRMSNorm(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 6 {
		return fmt.Errorf("router: RMSNorm expects 6 args, got %d", len(args))
	}

	config, err := castRMSNormConfig(configFields)

	if err != nil {
		return err
	}

	input, scale, output, err := castThreePointers(args[:3], "RMSNorm")

	if err != nil {
		return err
	}

	rows, err := castInt(args[3], "RMSNorm", "rows")

	if err != nil {
		return err
	}

	lastDim, err := castInt(args[4], "RMSNorm", "lastDim")

	if err != nil {
		return err
	}

	format, err := castDType(args[5], "RMSNorm", "format")

	if err != nil {
		return err
	}

	deviceBackend.RMSNorm(config, input, scale, output, rows, lastDim, format)

	return nil
}

func callAdaptiveRMSNorm(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 8 {
		return fmt.Errorf("router: AdaptiveRMSNorm expects 8 args, got %d", len(args))
	}

	config, err := castRMSNormConfig(configFields)

	if err != nil {
		return err
	}

	input, modulation, output, err := castThreePointers(args[:3], "AdaptiveRMSNorm")

	if err != nil {
		return err
	}

	rows, lastDim, rowsPerBatch, modulationCols, err := castFourInts(args[3:7], "AdaptiveRMSNorm")

	if err != nil {
		return err
	}

	format, err := castDType(args[7], "AdaptiveRMSNorm", "format")

	if err != nil {
		return err
	}

	deviceBackend.AdaptiveRMSNorm(
		config,
		input, modulation, output,
		rows, lastDim, rowsPerBatch, modulationCols,
		format,
	)

	return nil
}

func callLayerNorm(deviceBackend executionDevice, args []any) error {
	if len(args) != 7 {
		return fmt.Errorf("router: LayerNorm expects 7 args, got %d", len(args))
	}

	input, scale, bias, err := castThreePointers(args[:3], "LayerNorm")

	if err != nil {
		return err
	}

	output, err := castPointer(args[3], "LayerNorm", "output")

	if err != nil {
		return err
	}

	rows, err := castInt(args[4], "LayerNorm", "rows")

	if err != nil {
		return err
	}

	lastDim, err := castInt(args[5], "LayerNorm", "lastDim")

	if err != nil {
		return err
	}

	format, err := castDType(args[6], "LayerNorm", "format")

	if err != nil {
		return err
	}

	deviceBackend.LayerNorm(input, scale, bias, output, rows, lastDim, format)

	return nil
}

func callGroupNorm(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 8 {
		return fmt.Errorf("router: GroupNorm expects 8 args, got %d", len(args))
	}

	config, err := castGroupNormConfig(configFields)

	if err != nil {
		return err
	}

	input, scale, bias, output, err := castFourPointers(args[:4], "GroupNorm")

	if err != nil {
		return err
	}

	batch, channels, spatial, err := castThreeInts(args[4:7], "GroupNorm")

	if err != nil {
		return err
	}

	format, err := castDType(args[7], "GroupNorm", "format")

	if err != nil {
		return err
	}

	deviceBackend.GroupNorm(config, input, scale, bias, output, batch, channels, spatial, format)

	return nil
}

func callBinary(method string, args []any, call binaryDeviceCall) error {
	if len(args) != 5 {
		return fmt.Errorf("router: %s expects 5 args, got %d", method, len(args))
	}

	output, left, right, err := castThreePointers(args[:3], method)

	if err != nil {
		return err
	}

	count, err := castInt(args[3], method, "count")

	if err != nil {
		return err
	}

	format, err := castDType(args[4], method, "format")

	if err != nil {
		return err
	}

	call(output, left, right, count, format)

	return nil
}

func callUnary(method string, args []any, call unaryDeviceCall) error {
	if len(args) != 4 {
		return fmt.Errorf("router: %s expects 4 args, got %d", method, len(args))
	}

	output, err := castPointer(args[0], method, "output")

	if err != nil {
		return err
	}

	input, err := castPointer(args[1], method, "input")

	if err != nil {
		return err
	}

	count, err := castInt(args[2], method, "count")

	if err != nil {
		return err
	}

	format, err := castDType(args[3], method, "format")

	if err != nil {
		return err
	}

	call(output, input, count, format)

	return nil
}

func callSwiGLU(deviceBackend executionDevice, args []any) error {
	if len(args) != 5 {
		return fmt.Errorf("router: SwiGLU expects 5 args, got %d", len(args))
	}

	output, packed, err := castTwoPointers(args[:2], "SwiGLU")

	if err != nil {
		return err
	}

	batch, err := castInt(args[2], "SwiGLU", "batch")

	if err != nil {
		return err
	}

	halfCount, err := castInt(args[3], "SwiGLU", "halfCount")

	if err != nil {
		return err
	}

	format, err := castDType(args[4], "SwiGLU", "format")

	if err != nil {
		return err
	}

	deviceBackend.SwiGLU(output, packed, batch, halfCount, format)

	return nil
}

func callSwiGLUTensors(deviceBackend executionDevice, args []any) error {
	if len(args) != 5 {
		return fmt.Errorf("router: SwiGLUTensors expects 5 args, got %d", len(args))
	}

	output, gate, up, err := castThreePointers(args[:3], "SwiGLUTensors")

	if err != nil {
		return err
	}

	count, err := castInt(args[3], "SwiGLUTensors", "count")

	if err != nil {
		return err
	}

	format, err := castDType(args[4], "SwiGLUTensors", "format")

	if err != nil {
		return err
	}

	deviceBackend.SwiGLUTensors(output, gate, up, count, format)

	return nil
}

func callRoPE(deviceBackend executionDevice, configFields map[string]any, args []any) error {
	if len(args) != 6 {
		return fmt.Errorf("router: RoPE expects 6 args, got %d", len(args))
	}

	config, err := castRoPEConfig(configFields)

	if err != nil {
		return err
	}

	input, output, err := castTwoPointers(args[:2], "RoPE")

	if err != nil {
		return err
	}

	seqLen, numHeads, headDim, err := castThreeInts(args[2:5], "RoPE")

	if err != nil {
		return err
	}

	format, err := castDType(args[5], "RoPE", "format")

	if err != nil {
		return err
	}

	deviceBackend.RoPE(config, input, output, seqLen, numHeads, headDim, format)

	return nil
}

func callMultiHeadAttention(deviceBackend executionDevice, configFields map[string]any, args []any) error {
	if len(args) != 7 {
		return fmt.Errorf("router: MultiHeadAttention expects 7 args, got %d", len(args))
	}

	config, err := castMultiHeadAttentionConfig(configFields)

	if err != nil {
		return err
	}

	query, key, value, err := castThreePointers(args[:3], "MultiHeadAttention")

	if err != nil {
		return err
	}

	output, err := castPointer(args[3], "MultiHeadAttention", "output")

	if err != nil {
		return err
	}

	seqQ, err := castInt(args[4], "MultiHeadAttention", "seqQ")

	if err != nil {
		return err
	}

	seqK, err := castInt(args[5], "MultiHeadAttention", "seqK")

	if err != nil {
		return err
	}

	format, err := castDType(args[6], "MultiHeadAttention", "format")

	if err != nil {
		return err
	}

	deviceBackend.MultiHeadAttention(config, query, key, value, output, seqQ, seqK, format)

	return nil
}
