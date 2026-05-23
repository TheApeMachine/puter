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
	"prelu_slope", "leaky_relu_slope", "elu_alpha", "celu_alpha", "threshold",
	"snake", "hard_shrink", "soft_shrink", "hard_tanh_range", "snake_parametric",
	"rrelu", "prelu_v", "softmax",
	"glu", "geglu", "geglu_tanh", "swiglu", "reglu", "siglu", "linglu", "seglu",
}
