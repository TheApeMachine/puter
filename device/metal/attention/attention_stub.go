//go:build !darwin || !cgo

package attention

func (attention *Attention) stubHost() {
	attention.host.NeedsPlatform()
}
