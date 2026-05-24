package embedding

/*
Embedding implements device.Embedding for the CPU backend.
*/
type Embedding struct{}

/*
New constructs a Embedding receiver for CPU dispatch.
*/
func New() Embedding {
	return Embedding{}
}
