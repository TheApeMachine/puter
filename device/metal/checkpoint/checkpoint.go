package checkpoint

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Checkpoint implements device.Checkpoint for the Metal backend.
*/
type Checkpoint struct {
	host Host
}

/*
New wires a Checkpoint receiver to its Metal dispatch host.
*/
func New(host Host) Checkpoint {
	return Checkpoint{host: host}
}

/*
Host is the Metal dispatch surface checkpoint operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchCheckpointEncode(input, output unsafe.Pointer, format dtype.DType)
	DispatchCheckpointDecode(input, output unsafe.Pointer, format dtype.DType)
}
