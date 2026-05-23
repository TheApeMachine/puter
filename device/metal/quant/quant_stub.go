//go:build !darwin || !cgo

package quant

func (quantization *Quantization) stubHost() {
	quantization.host.NeedsPlatform()
}
