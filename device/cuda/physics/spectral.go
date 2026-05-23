//go:build cuda

package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (physics *Physics) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.host.DispatchFFT1D(realIn, imagIn, realOut, imagOut, count, format)
}

func (physics *Physics) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.host.DispatchIFFT1D(realIn, imagIn, realOut, imagOut, count, format)
}
