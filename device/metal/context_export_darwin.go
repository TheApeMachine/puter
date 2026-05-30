//go:build darwin && cgo

package metal

import "github.com/theapemachine/puter/device/metal/fusion"

/*
MetalContextRef returns the bridge device handle used for Metal dispatch.
*/
func (backend *Backend) MetalContextRef() uintptr {
	if backend == nil || backend.bridge == nil {
		return 0
	}

	return backend.bridge.contextRef()
}

/*
FusionCache returns the session fusion program cache.
*/
func (backend *Backend) FusionCache() *fusion.Cache {
	if backend == nil || backend.bridge == nil {
		return nil
	}

	return backend.bridge.fusionCache
}
