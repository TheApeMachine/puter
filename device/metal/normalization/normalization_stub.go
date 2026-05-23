//go:build !darwin || !cgo

package normalization

func (normalization *Normalization) stubHost() {
	normalization.host.NeedsPlatform()
}
