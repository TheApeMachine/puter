//go:build xla

package activation

/*
Lowering registers the activation-family XLA operation identifiers.
Program keys and HLO rendering live in device/xla/lowering_registry.go and
device/xla/internal/hlo.
*/
var LoweringOperations = []string{
	"relu", "exp", "log", "log1p", "expm1", "sigmoid", "log_sigmoid",
	"tanh", "silu", "swish", "gelu_tanh", "gelu", "leaky_relu", "elu",
	"celu", "selu", "softplus", "mish", "softsign", "hard_sigmoid",
	"hard_swish", "hard_tanh", "hard_gelu", "quick_gelu", "tanh_shrink",
}
