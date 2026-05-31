package model_editing

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
ModelEditing implements device.ModelEditing for the Metal backend.
*/
type ModelEditing struct {
	host Host
}

/*
New wires a ModelEditing receiver to its Metal dispatch host.
*/
func New(host Host) ModelEditing {
	return ModelEditing{host: host}
}

/*
Host is the Metal dispatch surface model-editing operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchWeightGraftAdd(weights, injection unsafe.Pointer, count int, format dtype.DType)
}
