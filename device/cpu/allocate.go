package cpu

import "unsafe"

const workspaceAlign = 64

func (backend *Backend) releaseWorkspace() {
	// Reserved for workspace slots allocated through allocateAligned.
}

func alignWorkspaceSize(size int64) int64 {
	if size <= 0 {
		return 0
	}

	return (size + workspaceAlign - 1) &^ (workspaceAlign - 1)
}

func (backend *Backend) allocateAligned(byteCount int64) (unsafe.Pointer, error) {
	return platformAllocateAligned(alignWorkspaceSize(byteCount))
}

func (backend *Backend) release(devicePointer unsafe.Pointer) {
	platformRelease(devicePointer)
}
