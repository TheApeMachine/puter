package activation

import "github.com/theapemachine/manifesto/dtype"

/*
LUTTable returns the production f16/bf16 LUT for a unary activation when one exists.
*/
func LUTTable(operation string, format dtype.DType) (*[65536]uint16, bool) {
	switch operation {
	case "exp":
		return lutPair(&expF16LUT, &expBF16LUT, format)
	case "log":
		return lutPair(&logF16LUT, &logBF16LUT, format)
	case "log1p":
		return lutPair(&log1pF16LUT, &log1pBF16LUT, format)
	case "expm1":
		return lutPair(&expm1F16LUT, &expm1BF16LUT, format)
	case "sigmoid":
		return lutPair(&sigmoidF16LUT, &sigmoidBF16LUT, format)
	case "log_sigmoid":
		return lutPair(&logSigmoidF16LUT, &logSigmoidBF16LUT, format)
	case "tanh":
		return lutPair(&tanhF16LUT, &tanhBF16LUT, format)
	case "silu", "swish":
		return lutPair(&siluF16LUT, &siluBF16LUT, format)
	case "gelu_tanh":
		return lutPair(&geluTanhF16LUT, &geluTanhBF16LUT, format)
	case "gelu":
		return lutPair(&geluF16LUT, &geluBF16LUT, format)
	case "relu":
		return lutPair(&reluF16LUT, &reluBF16LUT, format)
	case "leaky_relu":
		return lutPair(&leakyReluF16LUT, &leakyReluBF16LUT, format)
	case "elu":
		return lutPair(&eluF16LUT, &eluBF16LUT, format)
	case "celu":
		return lutPair(&celuF16LUT, &celuBF16LUT, format)
	case "selu":
		return lutPair(&seluF16LUT, &seluBF16LUT, format)
	case "softplus":
		return lutPair(&softplusF16LUT, &softplusBF16LUT, format)
	case "mish":
		return lutPair(&mishF16LUT, &mishBF16LUT, format)
	case "softsign":
		return lutPair(&softsignF16LUT, &softsignBF16LUT, format)
	case "hardsigmoid":
		return lutPair(&hardSigmoidF16LUT, &hardSigmoidBF16LUT, format)
	case "hardswish":
		return lutPair(&hardSwishF16LUT, &hardSwishBF16LUT, format)
	case "hardtanh":
		return lutPair(&hardTanhF16LUT, &hardTanhBF16LUT, format)
	case "hard_gelu":
		return lutPair(&hardGeluF16LUT, &hardGeluBF16LUT, format)
	case "quick_gelu":
		return lutPair(&quickGeluF16LUT, &quickGeluBF16LUT, format)
	case "tanh_shrink":
		return lutPair(&tanhShrinkF16LUT, &tanhShrinkBF16LUT, format)
	default:
		return nil, false
	}
}

func lutPair(
	f16Table *[65536]uint16,
	bf16Table *[65536]uint16,
	format dtype.DType,
) (*[65536]uint16, bool) {
	switch format {
	case dtype.Float16:
		return f16Table, true
	case dtype.BFloat16:
		return bf16Table, true
	default:
		return nil, false
	}
}
