package model_editing

/*
ModelEditing implements device.ModelEditing for the CPU backend.
*/
type ModelEditing struct{}

/*
New constructs a ModelEditing receiver for CPU dispatch.
*/
func New() ModelEditing {
	return ModelEditing{}
}
