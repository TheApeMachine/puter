//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal/normalization"
)

func (backend *Backend) GatedResidual(
	residual, branch, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols, set int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	if backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	if rowsPerBatch <= 0 || rows%rowsPerBatch != 0 {
		panic(tensor.ErrShapeMismatch)
	}

	if modulationCols < (set*3+3)*lastDim {
		panic(tensor.ErrShapeMismatch)
	}

	if err := normalization.DispatchGatedResidualRefs(
		backend.bridge.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(residual))),
		uintptr(unsafe.Pointer(resolveBufferRef(branch))),
		uintptr(unsafe.Pointer(resolveBufferRef(modulation))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows*lastDim),
		uint32(lastDim),
		uint32(rowsPerBatch),
		uint32(modulationCols),
		uint32(set),
	); err != nil {
		panic(err)
	}
}
