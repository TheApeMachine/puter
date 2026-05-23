//go:build xla

package reduction

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

func (reduction *Reduction) Sum(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.unimplemented("Sum")
	return 0
}

func (reduction *Reduction) Prod(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.unimplemented("Prod")
	return 0
}

func (reduction *Reduction) ReduceMin(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.unimplemented("ReduceMin")
	return 0
}

func (reduction *Reduction) ReduceMax(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.unimplemented("ReduceMax")
	return 0
}

func (reduction *Reduction) L1Norm(values unsafe.Pointer, count int, format dtype.DType,) float32 {
	reduction.unimplemented("L1Norm")
	return 0
}

