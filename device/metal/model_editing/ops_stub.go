//go:build !darwin || !cgo

package model_editing

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (modelEditing *ModelEditing) WeightGraftAdd(
	weights, injection unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	modelEditing.host.NeedsPlatform()
}
