//go:build !darwin || !cgo

package sampling

func (sampling *Sampling) stubHost() {
	sampling.host.NeedsPlatform()
}
