//go:build !cuda

package sampling

func (sampling *Sampling) stubHost() {
	sampling.host.NeedsPlatform()
}
