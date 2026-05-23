//go:build !darwin || !cgo

package layernorm

func (norm *Norm) stubHost() {
	norm.host.NeedsPlatform()
}
