//go:build !xla

package reduction

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

func (reduction *Reduction) Sum(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.stubHost()
	return 0
}

func (reduction *Reduction) Prod(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.stubHost()
	return 0
}

func (reduction *Reduction) ReduceMin(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.stubHost()
	return 0
}

func (reduction *Reduction) ReduceMax(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.stubHost()
	return 0
}

func (reduction *Reduction) L1Norm(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.stubHost()
	return 0
}

