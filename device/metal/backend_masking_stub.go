//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (backend *Backend) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	backend.maskingNeedsPlatform(input, mask, output, count, format)
}

func (backend *Backend) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	backend.maskingNeedsPlatform(output, nil, output, seqQ*seqK, format)
}

func (backend *Backend) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	backend.maskingNeedsPlatform(scores, slope, output, seqQ*seqK, format)
}

func (backend *Backend) maskingNeedsPlatform(
	first unsafe.Pointer,
	second unsafe.Pointer,
	third unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = first
	_ = second
	_ = third
	_ = count
	_ = format

	if backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}
