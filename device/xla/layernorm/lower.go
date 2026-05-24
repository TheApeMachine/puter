//go:build xla

package layernorm

/*
LoweringOperations lists layernorm-family XLA operation identifiers.
*/
var LoweringOperations = []string{"layer_norm", "rms_norm"}
