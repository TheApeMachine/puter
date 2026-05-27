//go:build darwin && cgo

package parity

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	cpunormalization "github.com/theapemachine/puter/device/cpu/normalization"
)

var referenceActivation = cpuactivation.New()
var referenceLayerNorm = cpulayernorm.New()
var referenceNormalization = cpunormalization.New()

/*
UnaryReference computes the CPU production reference for a unary activation.
*/
type UnaryReference func(dst, src unsafe.Pointer, count int)

/*
ReferenceReLU returns the production CPU reference kernel for ReLU.
*/
func ReferenceReLU(format dtype.DType) UnaryReference {
	return productionUnaryReference(referenceActivation.ReLU, format)
}

/*
ReferenceExp returns the production CPU reference kernel for Exp.
*/
func ReferenceExp(format dtype.DType) UnaryReference {
	return productionUnaryReference(referenceActivation.Exp, format)
}

/*
ReferenceGeluTanh returns the production CPU reference kernel for GeluTanh.
*/
func ReferenceGeluTanh(format dtype.DType) UnaryReference {
	return productionUnaryReference(referenceActivation.GeluTanh, format)
}

/*
ReferenceGelu returns the production CPU reference kernel for Gelu.
*/
func ReferenceGelu(format dtype.DType) UnaryReference {
	return productionUnaryReference(referenceActivation.Gelu, format)
}

func productionUnaryReference(
	kernel func(dst, src unsafe.Pointer, count int, format dtype.DType),
	format dtype.DType,
) UnaryReference {
	return func(dst, src unsafe.Pointer, count int) {
		kernel(dst, src, count, format)
	}
}

/*
ComputeUnaryReference runs the CPU reference into float32 lanes.
*/
func ComputeUnaryReference(
	source []float32,
	format dtype.DType,
	reference UnaryReference,
) []float32 {
	sourceBytes, err := encodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceBytes))
	reference(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceBytes[0]),
		len(source),
	)

	decoded, err := decodeVector(destinationBytes, format)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
GroupNormReference computes GroupNorm using the CPU production dispatcher.
*/
func GroupNormReference(
	input, scale, bias []float32,
	batch, channels, spatial, groups int,
	storageDType dtype.DType,
) []float32 {
	inputBytes, err := encodeVector(input, storageDType)

	if err != nil {
		panic(err)
	}

	scaleBytes, err := encodeVector(scale, storageDType)

	if err != nil {
		panic(err)
	}

	biasBytes, err := encodeVector(bias, storageDType)

	if err != nil {
		panic(err)
	}

	outputBytes := make([]byte, len(inputBytes))
	referenceNormalization.GroupNorm(
		device.GroupNormConfig{Groups: groups},
		unsafe.Pointer(&inputBytes[0]),
		unsafe.Pointer(&scaleBytes[0]),
		unsafe.Pointer(&biasBytes[0]),
		unsafe.Pointer(&outputBytes[0]),
		batch,
		channels,
		spatial,
		storageDType,
	)

	decoded, err := decodeVector(outputBytes, storageDType)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
LayerNormReference computes LayerNorm using the CPU production dispatcher.
*/
func LayerNormReference(
	input, scale, bias []float32,
	rows, cols int,
	storageDType dtype.DType,
) []float32 {
	inputBytes, err := encodeVector(input, storageDType)

	if err != nil {
		panic(err)
	}

	scaleBytes, err := encodeVector(scale, storageDType)

	if err != nil {
		panic(err)
	}

	biasBytes, err := encodeVector(bias, storageDType)

	if err != nil {
		panic(err)
	}

	outputBytes := make([]byte, len(inputBytes))
	referenceLayerNorm.LayerNorm(
		unsafe.Pointer(&inputBytes[0]),
		unsafe.Pointer(&scaleBytes[0]),
		unsafe.Pointer(&biasBytes[0]),
		unsafe.Pointer(&outputBytes[0]),
		rows,
		cols,
		storageDType,
	)

	decoded, err := decodeVector(outputBytes, storageDType)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
ModulatedLayerNormReference computes ModulatedLayerNorm using the CPU
production dispatcher.
*/
func ModulatedLayerNormReference(
	config device.ModulatedLayerNormConfig,
	input, modulation []float32,
	rows, cols, rowsPerBatch, modulationCols int,
	storageDType dtype.DType,
) []float32 {
	outputBytes := ModulatedLayerNormReferenceBytes(
		config,
		input,
		modulation,
		rows,
		cols,
		rowsPerBatch,
		modulationCols,
		storageDType,
	)

	decoded, err := decodeVector(outputBytes, storageDType)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
ModulatedLayerNormReferenceBytes returns the encoded CPU output.
*/
func ModulatedLayerNormReferenceBytes(
	config device.ModulatedLayerNormConfig,
	input, modulation []float32,
	rows, cols, rowsPerBatch, modulationCols int,
	storageDType dtype.DType,
) []byte {
	inputBytes, err := encodeVector(input, storageDType)

	if err != nil {
		panic(err)
	}

	modulationBytes, err := encodeVector(modulation, storageDType)

	if err != nil {
		panic(err)
	}

	outputBytes := make([]byte, len(inputBytes))
	referenceLayerNorm.ModulatedLayerNorm(
		config,
		unsafe.Pointer(&inputBytes[0]),
		unsafe.Pointer(&modulationBytes[0]),
		unsafe.Pointer(&outputBytes[0]),
		rows,
		cols,
		rowsPerBatch,
		modulationCols,
		storageDType,
	)

	return outputBytes
}

/*
ComputeUnaryReferenceBytes runs the CPU reference and returns encoded storage bytes.
*/
func ComputeUnaryReferenceBytes(
	source []float32,
	format dtype.DType,
	reference UnaryReference,
) []byte {
	sourceBytes, err := encodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceBytes))
	reference(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceBytes[0]),
		len(source),
	)

	return destinationBytes
}
