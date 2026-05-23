//go:build xla

package elementwise

/*
LoweringOperations lists elementwise-family XLA operation identifiers.
*/
var LoweringOperations = []string{
	"add", "sub", "mul", "div", "max", "min",
	"abs", "neg", "sqrt", "relu", "axpy",
}
