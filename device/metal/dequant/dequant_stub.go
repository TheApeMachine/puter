//go:build !darwin || !cgo

package dequant

func (dequantization *Dequantization) stubHost() {
	dequantization.host.NeedsPlatform()
}
