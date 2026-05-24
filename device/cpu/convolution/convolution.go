package convolution

/*
Convolution implements device.Convolution for the CPU backend.
*/
type Convolution struct{}

/*
New constructs a Convolution receiver for CPU dispatch.
*/
func New() Convolution {
	return Convolution{}
}
