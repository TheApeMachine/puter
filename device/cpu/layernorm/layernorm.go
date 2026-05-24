package layernorm

/*
Norm implements device.LayerNorm for the CPU backend.
*/
type Norm struct{}

/*
New constructs a Norm receiver for CPU dispatch.
*/
func New() Norm {
	return Norm{}
}
