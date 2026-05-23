//go:build !cuda

package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (physics *Physics) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.stubHost()
}

func (physics *Physics) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.stubHost()
}
