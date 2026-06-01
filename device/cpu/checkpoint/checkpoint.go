package checkpoint

/*
Checkpoint implements device.Checkpoint for the CPU backend.
*/
type Checkpoint struct{}

/*
New constructs a Checkpoint receiver for CPU dispatch.
*/
func New() Checkpoint {
	return Checkpoint{}
}
