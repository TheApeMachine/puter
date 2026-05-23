package activation

/*
StandardKernelName maps activation kernel enums to XLA lowering operation names.
*/
func StandardKernelName(kernel StandardKernel) (string, bool) {
	switch kernel {
	case StandardExp:
		return "exp", true
	case StandardLog:
		return "log", true
	case StandardLog1p:
		return "log1p", true
	case StandardExpm1:
		return "expm1", true
	case StandardSigmoid:
		return "sigmoid", true
	case StandardLogSigmoid:
		return "log_sigmoid", true
	case StandardTanh:
		return "tanh", true
	case StandardSilu:
		return "silu", true
	case StandardSwish:
		return "swish", true
	case StandardGeluTanh:
		return "gelu_tanh", true
	case StandardGelu:
		return "gelu", true
	case StandardReLU:
		return "relu", true
	case StandardLeakyReLU:
		return "leaky_relu", true
	case StandardELU:
		return "elu", true
	case StandardCELU:
		return "celu", true
	case StandardSELU:
		return "selu", true
	case StandardSoftplus:
		return "softplus", true
	case StandardMish:
		return "mish", true
	case StandardSoftsign:
		return "softsign", true
	case StandardHardSigmoid:
		return "hard_sigmoid", true
	case StandardHardSwish:
		return "hard_swish", true
	case StandardHardTanh:
		return "hard_tanh", true
	case StandardHardGelu:
		return "hard_gelu", true
	case StandardQuickGelu:
		return "quick_gelu", true
	case StandardTanhShrink:
		return "tanh_shrink", true
	default:
		return "", false
	}
}
