package masking

/*
Masking implements device.Masking for the CPU backend.
*/
type Masking struct{}

/*
New constructs a Masking receiver for CPU dispatch.
*/
func New() Masking {
	return Masking{}
}
