//go:build unix

package cpu

import (
	"sync"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
	"golang.org/x/sys/unix"
)

var unixMappings sync.Map

func platformAllocateAligned(byteCount int64) (unsafe.Pointer, error) {
	if byteCount <= 0 {
		return nil, nil
	}

	pageSize := unix.Getpagesize()
	mappedSize := int(((byteCount + int64(pageSize) - 1) / int64(pageSize)) * int64(pageSize))

	mapping, err := unix.Mmap(
		-1,
		0,
		mappedSize,
		unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_ANON|unix.MAP_PRIVATE,
	)

	if err != nil {
		return nil, tensor.ErrAllocatorExhausted
	}

	devicePointer := unsafe.Pointer(&mapping[0])
	unixMappings.Store(devicePointer, mapping)

	return devicePointer, nil
}

func platformRelease(devicePointer unsafe.Pointer) {
	if devicePointer == nil {
		return
	}

	mappingValue, loaded := unixMappings.LoadAndDelete(devicePointer)

	if !loaded {
		return
	}

	mapping := mappingValue.([]byte)
	_ = unix.Munmap(mapping)
}
