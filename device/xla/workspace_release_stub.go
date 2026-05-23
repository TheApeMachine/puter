//go:build !xla

package xla

func (backend *Backend) releaseWorkspace() {}
