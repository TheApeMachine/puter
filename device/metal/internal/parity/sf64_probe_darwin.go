//go:build darwin && cgo

package parity

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "sf64_probe.h"
#include "native/sf64_probe.m"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

/*
DispatchSF64TranscendentalProbe runs sf64_transcendental_probe on the Metal device.
*/
func (harness *Harness) DispatchSF64TranscendentalProbe(
	inputs []float32,
	sqrtInputs []uint64,
) ([]uint64, error) {
	caseCount := len(sqrtInputs)

	if caseCount == 0 {
		return nil, nil
	}

	if len(inputs) != caseCount*MetalSF64ProbeInputFloats {
		return nil, fmt.Errorf(
			"metal sf64 probe: input length %d != %d cases",
			len(inputs),
			caseCount,
		)
	}

	inputBuffer := harness.uploadFloat32(inputs)
	defer inputBuffer.Close()

	sqrtBuffer := harness.uploadUInt64(sqrtInputs)
	defer sqrtBuffer.Close()

	outputWords := caseCount * MetalSF64ProbeOutputWords
	outputBuffer := harness.uploadUInt64(make([]uint64, outputWords))
	defer outputBuffer.Close()

	var status C.MetalStatus
	code := C.metal_dispatch_sf64_transcendental_probe(
		harness.device,
		inputBuffer.buffer,
		sqrtBuffer.buffer,
		outputBuffer.buffer,
		C.uint32_t(caseCount),
		0,
		&status,
	)

	if code != 0 {
		return nil, fmt.Errorf(
			"metal sf64 probe dispatch failed (code=%d): %s",
			int(status.code),
			C.GoString(&status.message[0]),
		)
	}

	harness.Sync()

	return outputBuffer.downloadUInt64(), nil
}

const (
	MetalSF64ProbeOutputWords = int(C.MetalSF64ProbeOutputWords)
	MetalSF64ProbeInputFloats = int(C.MetalSF64ProbeInputFloats)
)

func (harness *Harness) uploadFloat32(values []float32) *Buffer {
	bytesIn := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(bytesIn[index*4:], mathFloat32Bits(value))
	}

	return harness.uploadBytes(bytesIn)
}

func (harness *Harness) uploadUInt64(values []uint64) *Buffer {
	bytesIn := make([]byte, len(values)*8)

	for index, value := range values {
		binary.LittleEndian.PutUint64(bytesIn[index*8:], value)
	}

	return harness.uploadBytes(bytesIn)
}

func (buffer *Buffer) downloadUInt64() []uint64 {
	bytesOut := buffer.readBytes()
	values := make([]uint64, len(bytesOut)/8)

	for index := range values {
		values[index] = binary.LittleEndian.Uint64(bytesOut[index*8:])
	}

	return values
}

func mathFloat32Bits(value float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&value))
}
