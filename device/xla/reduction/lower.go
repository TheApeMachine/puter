//go:build xla

package reduction

/*
LoweringOperations lists reduction-family XLA operation identifiers.
*/
var LoweringOperations = []string{
	"reduce_sum", "reduce_prod", "reduce_min", "reduce_max", "reduce_l1norm",
}
