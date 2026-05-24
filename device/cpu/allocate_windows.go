//go:build windows

package cpu

import (
	"sync"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
	"golang.org/x/sys/windows"
)

var windowsMappings sync.Map

func platformAllocateAligned(byteCount int64) (unsafe.Pointer, error) {
	if byteCount <= 0 {
		return nil, nil
	}

	address, err := windows.VirtualAlloc(
		0,
		uintptr(byteCount),
		windows.MEM_COMMIT|windows.MEM_RESERVE,
		windows.PAGE_READWRITE,
	)

	if err != nil {
		return nil, tensor.ErrAllocatorExhausted
	}

	devicePointer := unsafe.Pointer(address)
	windowsMappings.Store(devicePointer, address)

	return devicePointer, nil
}

func platformRelease(devicePointer unsafe.Pointer) {
	if devicePointer == nil {
		return
	}

	addressValue, loaded := windowsMappings.LoadAndDelete(devicePointer)

	if !loaded {
		return
	}

	address := addressValue.(uintptr)
	_ = windows.VirtualFree(address, 0, windows.MEM_RELEASE)
}
