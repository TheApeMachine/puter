//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

/*
Resident returns the canonical data pointer for a Metal-resident tensor.
device.Backend methods take these pointers; they must come from tensors
allocated through this backend's Upload or NewZeroed.
*/
func Resident(value tensor.Tensor) unsafe.Pointer {
	target, err := requireMetalTensor(value)

	if err != nil {
		return nil
	}

	return target.residentPointer()
}

func (target *metalTensor) residentPointer() unsafe.Pointer {
	if target == nil || target.buffer == nil || target.bytes == 0 {
		return nil
	}

	return metalBufferContents(unsafe.Pointer(target.buffer))
}

func (bridge *metalBridge) registerResident(target *metalTensor) {
	pointer := target.residentPointer()

	if pointer == nil {
		return
	}

	bridge.resident.Store(pointer, target)
}

func (bridge *metalBridge) unregisterResident(target *metalTensor) {
	pointer := target.residentPointer()

	if pointer == nil {
		return
	}

	bridge.resident.Delete(pointer)
}

func (bridge *metalBridge) tensorAt(pointer unsafe.Pointer) (*metalTensor, error) {
	if pointer == nil {
		return nil, tensor.ErrShapeMismatch
	}

	value, ok := bridge.resident.Load(pointer)

	if !ok {
		return nil, errResidentPointer
	}

	target, ok := value.(*metalTensor)

	if !ok || target == nil {
		return nil, errResidentPointer
	}

	return target, nil
}

func (backend *Backend) tensorAt(pointer unsafe.Pointer) (*metalTensor, error) {
	if backend == nil || backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.tensorAt(pointer)
}

func (backend *Backend) tensorsAt(pointers ...unsafe.Pointer) ([]*metalTensor, error) {
	tensors := make([]*metalTensor, len(pointers))

	for index, pointer := range pointers {
		target, err := backend.tensorAt(pointer)

		if err != nil {
			return nil, err
		}

		tensors[index] = target
	}

	return tensors, nil
}

var errResidentPointer = tensor.ErrShapeMismatch
