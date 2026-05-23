//go:build !cuda

package quant

func (quantization *Quantization) stubHost() {
	quantization.host.NeedsPlatform()
}
