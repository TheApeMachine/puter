//go:build !darwin || !cgo

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (reduction *Reduction) Sum(dst, values unsafe.Pointer, count int, format dtype.DType) {
	reduction.stubHost()
}

func (reduction *Reduction) Prod(dst, values unsafe.Pointer, count int, format dtype.DType) {
	reduction.stubHost()
}

func (reduction *Reduction) ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType) {
	reduction.stubHost()
}

func (reduction *Reduction) ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType) {
	reduction.stubHost()
}

func (reduction *Reduction) L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType) {
	reduction.stubHost()
}
