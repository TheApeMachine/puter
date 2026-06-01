package checkpoint

import (
	"encoding/binary"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func (checkpoint Checkpoint) CheckpointEncode(input, output unsafe.Pointer, format dtype.DType) {
	dispatch.RequireFloat32(format)

	inputData, inputCount, dims, wrapped := dispatch.ResolvePointer(input)

	if !wrapped {
		panic("checkpoint: encode requires dispatch.View on input")
	}

	outputData, outputCount, _, outputWrapped := dispatch.ResolvePointer(output)

	if !outputWrapped {
		panic("checkpoint: encode requires dispatch.View on output")
	}

	headerBytes := 16 + len(dims)*8
	dataBytes := inputCount * 4
	totalBytes := headerBytes + dataBytes

	if outputCount != totalBytes {
		panic("checkpoint: output byte length mismatch")
	}

	inputSlice := dispatch.Float32Slice(inputData, inputCount)
	outputSlice := dispatch.Uint8Slice(outputData, totalBytes)

	binary.LittleEndian.PutUint64(outputSlice[0:8], uint64(len(dims)))
	binary.LittleEndian.PutUint64(outputSlice[8:16], uint64(dataBytes))

	for index, dimension := range dims {
		binary.LittleEndian.PutUint64(outputSlice[16+index*8:], uint64(dimension))
	}

	EncodeFloat32DataNative(outputSlice[headerBytes:], inputSlice)
}

func (checkpoint Checkpoint) CheckpointDecode(input, output unsafe.Pointer, format dtype.DType) {
	dispatch.RequireFloat32(format)

	inputData, inputCount, _, inputWrapped := dispatch.ResolvePointer(input)

	if !inputWrapped {
		panic("checkpoint: decode requires dispatch.View on input")
	}

	inputSlice := dispatch.Uint8Slice(inputData, inputCount)

	if len(inputSlice) < 16 {
		panic("checkpoint: truncated header")
	}

	rank := int(binary.LittleEndian.Uint64(inputSlice[0:8]))
	dataBytes := int(binary.LittleEndian.Uint64(inputSlice[8:16]))
	headerBytes := 16 + rank*8

	if len(inputSlice) != headerBytes+dataBytes {
		panic("checkpoint: payload length mismatch")
	}

	elementCount := dataBytes / 4
	outputData, outputCount, _, outputWrapped := dispatch.ResolvePointer(output)

	if !outputWrapped {
		panic("checkpoint: decode requires dispatch.View on output")
	}

	if outputCount != elementCount {
		panic("checkpoint: output element count mismatch")
	}

	outputSlice := dispatch.Float32Slice(outputData, elementCount)

	DecodeFloat32DataNative(outputSlice, inputSlice[headerBytes:])
}
