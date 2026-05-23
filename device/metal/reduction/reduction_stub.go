//go:build !darwin || !cgo

package reduction

func (reduction *Reduction) stubHost() {
	reduction.host.NeedsPlatform()
}
