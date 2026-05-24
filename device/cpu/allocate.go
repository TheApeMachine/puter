package cpu

import "unsafe"

const workspaceAlign = 64

func (backend *Backend) releaseWorkspace() {
	backend.workspaceMu.Lock()
	blocks := backend.workspaceBlocks
	backend.workspaceBlocks = nil
	backend.workspaceMu.Unlock()

	for _, block := range blocks {
		backend.release(block)
	}
}

func alignWorkspaceSize(size int64) int64 {
	if size <= 0 {
		return 0
	}

	return (size + workspaceAlign - 1) &^ (workspaceAlign - 1)
}

func (backend *Backend) allocateAligned(byteCount int64) (unsafe.Pointer, error) {
	pointer, err := platformAllocateAligned(alignWorkspaceSize(byteCount))

	if pointer == nil || err != nil {
		return pointer, err
	}

	backend.workspaceMu.Lock()
	backend.workspaceBlocks = append(backend.workspaceBlocks, pointer)
	backend.workspaceMu.Unlock()

	return pointer, err
}

func (backend *Backend) release(devicePointer unsafe.Pointer) {
	platformRelease(devicePointer)
}
