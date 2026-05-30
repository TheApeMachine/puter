//go:build darwin && cgo

package fusion

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "fusion_jit.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

type programHandle struct {
	contextRef uintptr
	native     C.MetalFusionProgramRef
}

/*
Dispatch runs the fusion kernel against device-resident Metal buffers.
Each buffer reference must come from metal.BufferRefFromDispatch.
*/
func (program *Program) Dispatch(
	contextRef uintptr,
	inputBufferRefs []uintptr,
	outputBufferRef uintptr,
	count int,
) error {
	if program == nil {
		return fmt.Errorf("metal fusion: program is nil")
	}

	if count == 0 {
		return nil
	}

	if contextRef == 0 {
		return tensor.ErrNeedsPlatformSetup
	}

	if err := program.ensureCompiled(contextRef); err != nil {
		return err
	}

	inputRefs := make([]C.MetalBufferRef, len(inputBufferRefs))

	for inputIndex, bufferRef := range inputBufferRefs {
		if bufferRef == 0 {
			return fmt.Errorf("metal fusion: input %d is not a device buffer", inputIndex)
		}

		inputRefs[inputIndex] = (C.MetalBufferRef)(unsafe.Pointer(bufferRef))
	}

	if outputBufferRef == 0 {
		return fmt.Errorf("metal fusion: output is not a device buffer")
	}

	outputRef := (C.MetalBufferRef)(unsafe.Pointer(outputBufferRef))

	var inputSlot *C.MetalBufferRef

	if len(inputRefs) > 0 {
		inputSlot = &inputRefs[0]
	}

	status := C.MetalStatus{}
	code := C.metal_fusion_program_dispatch(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		program.handle.native,
		inputSlot,
		outputRef,
		C.int(len(inputRefs)),
		C.uint32_t(count),
		&status,
	)

	if code != 0 {
		return fmt.Errorf("metal fusion: dispatch: %s", C.GoString(&status.message[0]))
	}

	return nil
}

func (program *Program) ensureCompiled(contextRef uintptr) error {
	if program.handle.native != nil && program.handle.contextRef == contextRef {
		return nil
	}

	program.releaseHandle()

	sourceCString := C.CString(program.source)
	kernelCString := C.CString(program.kernelName)

	defer C.free(unsafe.Pointer(sourceCString))
	defer C.free(unsafe.Pointer(kernelCString))

	status := C.MetalStatus{}
	native := C.metal_fusion_program_compile(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		sourceCString,
		kernelCString,
		&status,
	)

	if native == nil {
		return fmt.Errorf(
			"metal fusion: compile %q: %s",
			program.kernelName,
			C.GoString(&status.message[0]),
		)
	}

	program.handle = programHandle{
		contextRef: contextRef,
		native:     native,
	}

	return nil
}

func (program *Program) releaseHandle() {
	if program == nil || program.handle.native == nil {
		return
	}

	C.metal_fusion_program_release(program.handle.native)
	program.handle.native = nil
	program.handle.contextRef = 0
}
