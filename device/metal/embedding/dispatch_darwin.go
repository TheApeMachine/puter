//go:build darwin && cgo

package embedding

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "lookup.h"
*/
import "C"

/*
elementDType maps a manifesto dtype identifier to the C enum the
embedding kernel suffix selector understands. The Metal embedding kernels
are compiled per dtype (embedding_lookup_float32, ..._float16,
..._bfloat16); anything outside that set is rejected here so the dispatch
path fails loudly rather than silently picking a wrong kernel.
*/
func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.MetalElementDTypeFloat32
	case dtype.Float16:
		return C.MetalElementDTypeFloat16
	case dtype.BFloat16:
		return C.MetalElementDTypeBFloat16
	default:
		return -1
	}
}

/*
DispatchLookup launches the metal_dispatch_embedding_lookup kernel with
typed C buffer references. Callers that have raw uintptrs (as produced by
the dispatcher's unsafe.Pointer plumbing) should use DispatchLookupRefs
which round-trips through the same path with the conversion done inline.
*/
func DispatchLookup(
	contextRef C.MetalDeviceRef,
	tableBuffer C.MetalBufferRef,
	indicesBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	vocab uint32,
	hidden uint32,
	indexCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_embedding_lookup(
		contextRef,
		elementFormat,
		tableBuffer,
		indicesBuffer,
		outBuffer,
		C.uint32_t(vocab),
		C.uint32_t(hidden),
		C.uint32_t(indexCount),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

/*
DispatchLookupRefs is the uintptr entry point the runtime dispatcher
calls. The dispatcher works in unsafe.Pointer values it carries through
its plan walk; the Metal compute host narrows them to uintptr and hands
them here so this layer can do the final C type conversions in one
place. Mirrors device/metal/matmul/dispatch_darwin.go::DispatchMatmulRefs.
*/
func DispatchLookupRefs(
	contextRef uintptr,
	tableBuffer uintptr,
	indicesBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	vocab uint32,
	hidden uint32,
	indexCount uint32,
) error {
	return DispatchLookup(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(tableBuffer)),
		C.MetalBufferRef(unsafe.Pointer(indicesBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		vocab,
		hidden,
		indexCount,
	)
}

var errUnsupportedDType = errors.New("metal embedding: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
