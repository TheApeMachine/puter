package masking

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Masking implements device.Masking for the Metal backend.
*/
type Masking struct {
	host Host
}

/*
New wires a Masking receiver to its Metal dispatch host.
*/
func New(host Host) Masking {
	return Masking{host: host}
}

/*
Host is the Metal dispatch surface masking operations call into.
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
