package model_editing

import cpumodelediting "github.com/theapemachine/puter/device/cpu/model_editing"

/*
ModelEditing implements device.ModelEditing on CUDA by delegating to the
CPU scalar reference until dedicated device paths land.
*/
type ModelEditing struct {
	cpumodelediting.ModelEditing
}

func New() ModelEditing {
	return ModelEditing{ModelEditing: cpumodelediting.New()}
}
