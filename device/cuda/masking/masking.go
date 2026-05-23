package masking

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Masking implements device.Masking for the CUDA backend.
Dispatch lives in the attention quintet on Metal; CUDA exposes the same
interface surface through this family type wired to the root ComputeHost.
*/
type Masking struct {
	host Host
}

/*
New wires a Masking receiver to its CUDA dispatch host.
*/
func New(host Host) Masking {
	return Masking{host: host}
}

/*
Host is the CUDA dispatch surface masking operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType)
	DispatchCausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
	DispatchALiBiBias(
		scores, slope, output unsafe.Pointer,
		seqQ, seqK int,
		format dtype.DType,
	)
}

func (masking *Masking) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	masking.host.DispatchApplyMask(input, mask, output, count, format)
}

func (masking *Masking) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.DispatchCausalMask(output, seqQ, seqK, format)
}

func (masking *Masking) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.DispatchALiBiBias(scores, slope, output, seqQ, seqK, format)
}
