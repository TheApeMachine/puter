//go:build !darwin || !cgo

package convolution

func (convolution *Convolution) stubHost() {
	convolution.host.NeedsPlatform()
}
