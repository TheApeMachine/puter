package interpretability

/*
Interpretability implements device.Interpretability on CPU.
*/
type Interpretability struct{}

/*
New constructs the CPU interpretability receiver.
*/
func New() Interpretability {
	return Interpretability{}
}
