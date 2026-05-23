//go:build darwin && cgo

package parity

import (
	"os"
	"path/filepath"
	"runtime"
)

func loadKernelsMetalLibrary() ([]byte, error) {
	_, sourcePath, _, ok := runtime.Caller(0)

	if !ok {
		return nil, os.ErrNotExist
	}

	libraryPath := filepath.Clean(filepath.Join(filepath.Dir(sourcePath), "..", "..", "kernels.metallib"))

	return os.ReadFile(libraryPath)
}
