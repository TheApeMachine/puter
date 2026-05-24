package sampling

/*
Sampling implements device.Sampling for the CPU backend.
*/
type Sampling struct{}

/*
New constructs a Sampling receiver for CPU dispatch.
*/
func New() Sampling {
	return Sampling{}
}

var Default = New()
