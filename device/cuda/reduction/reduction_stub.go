//go:build !cuda

package reduction

func (reduction *Reduction) stubHost() {
	reduction.host.NeedsPlatform()
}
