package xla

/*
NewRuntimeBuilder constructs the shared XLA builder used by ComputeHost.
*/
func NewRuntimeBuilder() *Builder {
	return NewDefaultBuilder(DefaultBuilderTarget)
}
