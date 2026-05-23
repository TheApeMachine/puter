//go:build !cuda

package layernorm

func (norm *Norm) stubHost() {
	norm.host.NeedsPlatform()
}
