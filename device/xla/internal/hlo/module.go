package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
ModuleBuilder renders textual HLO for compile-cache entries.
*/
type ModuleBuilder struct {
	moduleName  string
	elementType string
	dimensions  []int64
}

/*
NewModuleBuilder constructs an HLO module builder for a dense output shape.
*/
func NewModuleBuilder(
	moduleName string,
	elementFormat dtype.DType,
	outputShape tensor.Shape,
) (*ModuleBuilder, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return nil, err
	}

	dimensions := make([]int64, len(outputShape.Dims()))

	for index, dimension := range outputShape.Dims() {
		dimensions[index] = int64(dimension)
	}

	return &ModuleBuilder{
		moduleName:  moduleName,
		elementType: elementType,
		dimensions:  dimensions,
	}, nil
}

/*
RenderUnary builds HLO for a single-input unary operation.
*/
func (moduleBuilder *ModuleBuilder) RenderUnary(operationName string, floatParams []float64) (string, error) {
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()

	switch operationName {
	case "relu":
		return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  p0 = %s parameter(0)
  zero = %s[] constant(0)
  bcast = %s broadcast(zero), dimensions={}
  ROOT result = %s maximum(p0, bcast)
}
`, moduleBuilder.moduleName, entryLayout, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral), nil
	case "exp":
		return moduleBuilder.unaryRoot("exponential", "p0"), nil
	case "log":
		return moduleBuilder.unaryRoot("log", "p0"), nil
	case "log1p":
		return moduleBuilder.unaryRoot("log1p", "p0"), nil
	case "expm1":
		return moduleBuilder.unaryRoot("expm1", "p0"), nil
	case "sigmoid", "logistic":
		return moduleBuilder.unaryRoot("logistic", "p0"), nil
	case "log_sigmoid":
		return moduleBuilder.renderLogSigmoid()
	case "tanh":
		return moduleBuilder.unaryRoot("tanh", "p0"), nil
	case "abs":
		return moduleBuilder.unaryRoot("abs", "p0"), nil
	case "neg", "negate":
		return moduleBuilder.unaryRoot("negate", "p0"), nil
	case "sqrt":
		return moduleBuilder.unaryRoot("sqrt", "p0"), nil
	case "silu", "swish":
		return moduleBuilder.renderSilu(), nil
	case "gelu":
		return moduleBuilder.renderGeluErf(), nil
	case "gelu_tanh":
		return moduleBuilder.renderGeluTanh(), nil
	case "elu":
		return moduleBuilder.renderELU(eluAlpha), nil
	case "celu":
		return moduleBuilder.renderCELU(celuAlpha), nil
	case "selu":
		return moduleBuilder.renderSELU(), nil
	case "softplus":
		return moduleBuilder.renderSoftplus(), nil
	case "mish":
		return moduleBuilder.renderMish(), nil
	case "softsign":
		return moduleBuilder.renderSoftsign(), nil
	case "hard_sigmoid":
		return moduleBuilder.renderHardSigmoid(), nil
	case "hard_swish":
		return moduleBuilder.renderHardSwish(), nil
	case "hard_tanh":
		return moduleBuilder.renderHardTanh(), nil
	case "hard_gelu":
		return moduleBuilder.renderHardGelu(), nil
	case "quick_gelu":
		return moduleBuilder.renderQuickGelu(), nil
	case "tanh_shrink":
		return moduleBuilder.renderTanhShrink(), nil
	case "leaky_relu":
		return moduleBuilder.renderLeakyReLU(floatParams)
	default:
		return "", fmt.Errorf("unsupported unary HLO operation: %s", operationName)
	}
}

/*
RenderBinary builds HLO for a two-input binary operation with NumPy broadcast.
*/
func (moduleBuilder *ModuleBuilder) RenderBinary(operationName string) (string, error) {
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()
	hloOp := binaryHLOOp(operationName)

	if hloOp == "" {
		return "", fmt.Errorf("unsupported binary HLO operation: %s", operationName)
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  p0 = %s parameter(0)
  p1 = %s parameter(1)
  ROOT result = %s %s(p0, p1)
}
`, moduleBuilder.moduleName, entryLayout, shapeLiteral, shapeLiteral, shapeLiteral, hloOp), nil
}

func (moduleBuilder *ModuleBuilder) unaryRoot(hloOp, parameterName string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  p0 = %s parameter(0)
  ROOT result = %s %s(p0)
}
`, moduleBuilder.moduleName, entryLayout, shapeLiteral, shapeLiteral, hloOp)
}

func (moduleBuilder *ModuleBuilder) renderLogSigmoid() (string, error) {
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  p0 = %s parameter(0)
  neg = %s negate(p0)
  exp = %s exponential(neg)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  denom = %s add(one_b, exp)
  ROOT result = %s subtract(p0, log(denom))
}
`, moduleBuilder.moduleName, entryLayout, shapeLiteral, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral), nil
}

func (moduleBuilder *ModuleBuilder) renderLeakyReLU(floatParams []float64) (string, error) {
	if len(floatParams) == 0 {
		return "", fmt.Errorf("leaky_relu requires negative slope parameter")
	}

	slope := floatParams[0]
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  p0 = %s parameter(0)
  zero = %s[] constant(0)
  pos = %s maximum(p0, zero)
  neg = %s minimum(p0, zero)
  slope = %s[] constant(%g)
  slope_b = %s broadcast(slope), dimensions={}
  scaled = %s multiply(neg, slope_b)
  ROOT result = %s add(pos, scaled)
}
`, moduleBuilder.moduleName, entryLayout, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, moduleBuilder.elementType, slope, shapeLiteral, shapeLiteral, shapeLiteral), nil
}

func (moduleBuilder *ModuleBuilder) shapeLiteral() string {
	if len(moduleBuilder.dimensions) == 0 {
		return fmt.Sprintf("%s[]", moduleBuilder.elementType)
	}

	dimensionText := make([]string, len(moduleBuilder.dimensions))

	for index, dimension := range moduleBuilder.dimensions {
		dimensionText[index] = fmt.Sprintf("%d", dimension)
	}

	layout := moduleBuilder.minorToMajorLayout()
	return fmt.Sprintf("%s[%s]{%s}", moduleBuilder.elementType, strings.Join(dimensionText, ","), layout)
}

func (moduleBuilder *ModuleBuilder) entryLayout() string {
	return moduleBuilder.shapeLiteral()
}

func (moduleBuilder *ModuleBuilder) minorToMajorLayout() string {
	rank := len(moduleBuilder.dimensions)
	indices := make([]string, rank)

	for index := range moduleBuilder.dimensions {
		indices[index] = fmt.Sprintf("%d", rank-1-index)
	}

	return strings.Join(indices, ",")
}

func elementToken(elementFormat dtype.DType) (string, error) {
	switch elementFormat {
	case dtype.Float64:
		return "f64", nil
	case dtype.Float32:
		return "f32", nil
	case dtype.Float16:
		return "f16", nil
	case dtype.BFloat16:
		return "bf16", nil
	case dtype.Float8E4M3:
		return "f8e4m3fn", nil
	case dtype.Float8E5M2:
		return "f8e5m2", nil
	case dtype.Int64:
		return "s64", nil
	case dtype.Int32:
		return "s32", nil
	case dtype.Int16:
		return "s16", nil
	case dtype.Int8:
		return "s8", nil
	case dtype.Uint64:
		return "u64", nil
	case dtype.Uint32:
		return "u32", nil
	case dtype.Uint16:
		return "u16", nil
	case dtype.Uint8:
		return "u8", nil
	case dtype.Bool:
		return "pred", nil
	default:
		return "", fmt.Errorf("unsupported HLO dtype: %v", elementFormat)
	}
}

func binaryHLOOp(operationName string) string {
	switch operationName {
	case "add":
		return "add"
	case "sub":
		return "subtract"
	case "mul":
		return "multiply"
	case "div":
		return "divide"
	case "max":
		return "maximum"
	case "min":
		return "minimum"
	default:
		return ""
	}
}
