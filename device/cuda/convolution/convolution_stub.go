//go:build !cuda

package convolution

func (convolution *Convolution) stubHost() {
	convolution.host.NeedsPlatform()
}
