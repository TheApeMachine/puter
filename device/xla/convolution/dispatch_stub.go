//go:build !xla

package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (convolution *Convolution) Conv2D( config Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType, ) {
	convolution.stubHost()
}

func (convolution *Convolution) Conv1D( config Conv1DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inLength, outChannels, kernelLength, outLength int, format dtype.DType, ) {
	convolution.stubHost()
}

func (convolution *Convolution) Conv3D( config Conv3DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW int, format dtype.DType, ) {
	convolution.stubHost()
}

func (convolution *Convolution) ConvTranspose2D( config Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType, ) {
	convolution.stubHost()
}

