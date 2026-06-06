package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation Activation) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardGeluF16LUT, &hardGeluBF16LUT, runHardGeluF32)
}

func (activation Activation) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &quickGeluF16LUT, &quickGeluBF16LUT, runQuickGeluF32)
}

func (activation Activation) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &tanhShrinkF16LUT, &tanhShrinkBF16LUT, runTanhShrinkF32)
}
