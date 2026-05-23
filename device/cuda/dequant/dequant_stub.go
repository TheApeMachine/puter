//go:build !cuda

package dequant

func (dequantization *Dequantization) stubHost() {
	dequantization.host.NeedsPlatform()
}
