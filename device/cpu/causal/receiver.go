package causal

/*
Causal implements device.Causal for the CPU backend.
*/
type Causal struct{}

/*
New constructs a Causal receiver for CPU dispatch.
*/
func New() Causal {
	return Causal{}
}
