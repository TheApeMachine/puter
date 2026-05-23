//go:build !cuda

package attention

func (attention *Attention) stubHost() {
	attention.host.NeedsPlatform()
}
