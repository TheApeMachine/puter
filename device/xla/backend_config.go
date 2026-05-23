package xla

/*
DefaultConfig returns the default XLA backend configuration.
*/
func DefaultConfig() map[string]any {
	return map[string]any{
		"target": DefaultBuilderTarget,
	}
}
