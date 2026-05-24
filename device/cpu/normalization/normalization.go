package normalization

/*
Normalization implements device.Normalization for the CPU backend.
*/
type Normalization struct{}

/*
New constructs a Normalization receiver for CPU dispatch.
*/
func New() Normalization {
	return Normalization{}
}

var Default = New()
