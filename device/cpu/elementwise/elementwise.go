package elementwise

/*
Elementwise implements device.Elementwise for the CPU backend.
*/
type Elementwise struct{}

/*
New constructs a Elementwise receiver for CPU dispatch.
*/
func New() Elementwise {
	return Elementwise{}
}
