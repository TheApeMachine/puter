package reduction

/*
Reduction implements device.Reduction for the CPU backend.
*/
type Reduction struct{}

/*
New constructs a Reduction receiver for CPU dispatch.
*/
func New() Reduction {
	return Reduction{}
}
