//go:build !cuda

package normalization

func (normalization *Normalization) stubHost() {
	normalization.host.NeedsPlatform()
}
