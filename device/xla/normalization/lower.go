//go:build xla

package normalization

/*
LoweringOperations lists normalization-family XLA operation identifiers.
*/
var LoweringOperations = []string{"batch_norm_eval", "instance_norm", "group_norm"}
