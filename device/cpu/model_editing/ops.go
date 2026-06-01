package model_editing

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func (modelEditing ModelEditing) WeightGraftAdd(
	weights, injection unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatch.RequireFloat32(format)

	if count == 0 {
		return
	}

	weightsData, _, _, _ := dispatch.ResolvePointer(weights)
	injectionData, _, _, _ := dispatch.ResolvePointer(injection)

	WeightGraftAddFloat32Native(
		dispatch.Float32Slice(weightsData, count),
		dispatch.Float32Slice(injectionData, count),
	)
}
