//go:build xla

package xla

/*
releaseWorkspace closes all PJRT-backed tensors referenced by workspace slots.
*/
func (backend *Backend) releaseWorkspace() {
	if backend.workspace == nil {
		return
	}

	backend.workspace.mutex.Lock()
	defer backend.workspace.mutex.Unlock()

	for _, residentPointer := range backend.workspace.slots {
		deviceTensor := resolveDeviceTensor(residentPointer)

		if deviceTensor != nil {
			_ = deviceTensor.Close()
		}
	}
}
