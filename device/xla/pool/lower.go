//go:build xla

package pool

/*
LoweringOperations lists pool-family XLA operation identifiers.
*/
var LoweringOperations = []string{
	"max_pool2d", "avg_pool2d", "adaptive_max_pool2d", "adaptive_avg_pool2d",
}
