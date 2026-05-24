package activation

/*
Activation implements device.Activation for the CPU backend.
*/
type Activation struct{}

/*
New constructs a Activation receiver for CPU dispatch.
*/
func New() Activation {
	return Activation{}
}
