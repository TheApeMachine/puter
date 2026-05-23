package hlo

import (
	"fmt"
	"math"
)

func (moduleBuilder *ModuleBuilder) renderUnaryParamOp(operationName string, floatParams []float64) (string, error) {
	switch operationName {
	case "prelu_slope", "leaky_relu_slope":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("%s requires slope parameter", operationName)
		}

		return moduleBuilder.renderLeakyReLU(floatParams)
	case "elu_alpha":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("elu_alpha requires alpha parameter")
		}

		return moduleBuilder.renderELU(floatParams[0]), nil
	case "celu_alpha":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("celu_alpha requires alpha parameter")
		}

		return moduleBuilder.renderCELU(floatParams[0]), nil
	case "threshold":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("threshold requires threshold parameter")
		}

		return moduleBuilder.renderThreshold(floatParams[0]), nil
	case "snake":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("snake requires alpha parameter")
		}

		return moduleBuilder.renderSnake(floatParams[0]), nil
	case "hard_shrink":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("hard_shrink requires lambda parameter")
		}

		return moduleBuilder.renderHardShrink(floatParams[0]), nil
	case "soft_shrink":
		if len(floatParams) == 0 {
			return "", fmt.Errorf("soft_shrink requires lambda parameter")
		}

		return moduleBuilder.renderSoftShrink(floatParams[0]), nil
	default:
		return "", fmt.Errorf("unsupported unary param HLO operation: %s", operationName)
	}
}

func (moduleBuilder *ModuleBuilder) renderDualParamOp(operationName string, floatParams []float64) (string, error) {
	if len(floatParams) < 2 {
		return "", fmt.Errorf("%s requires two float parameters", operationName)
	}

	switch operationName {
	case "hard_tanh_range":
		return moduleBuilder.renderHardTanhRange(floatParams[0], floatParams[1]), nil
	case "snake_parametric":
		return moduleBuilder.renderSnakeParametric(floatParams[0], floatParams[1]), nil
	case "rrelu":
		return moduleBuilder.renderRReLU(float32(floatParams[0]), float32(floatParams[1])), nil
	default:
		return "", fmt.Errorf("unsupported dual param HLO operation: %s", operationName)
	}
}

func (moduleBuilder *ModuleBuilder) renderThreshold(threshold float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  threshold = %s[] constant(%g)
  threshold_b = %s broadcast(threshold), dimensions={}
  pred = %s compare(p0, threshold_b), direction=GT
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  ROOT result = %s select(pred, p0, zero_b)
`, shapeLiteral, moduleBuilder.elementType, threshold, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSnake(alpha float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  scaled = %s multiply(p0, alpha_b)
  sine = %s sine(scaled)
  sq = %s multiply(sine, sine)
  inv_alpha = %s[] constant(%g)
  inv_alpha_b = %s broadcast(inv_alpha), dimensions={}
  addend = %s multiply(inv_alpha_b, sq)
  ROOT result = %s add(p0, addend)
`, shapeLiteral, moduleBuilder.elementType, alpha, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, 1.0/alpha, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardShrink(lambda float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  abs_val = %s abs(p0)
  lambda = %s[] constant(%g)
  lambda_b = %s broadcast(lambda), dimensions={}
  pred = %s compare(abs_val, lambda_b), direction=GT
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  ROOT result = %s select(pred, p0, zero_b)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, lambda, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSoftShrink(lambda float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  lambda = %s[] constant(%g)
  lambda_b = %s broadcast(lambda), dimensions={}
  pos = %s compare(p0, lambda_b), direction=GT
  neg = %s compare(p0, negate(lambda_b)), direction=LT
  pos_val = %s subtract(p0, lambda_b)
  neg_val = %s add(p0, lambda_b)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pos_sel = %s select(pos, pos_val, zero_b)
  ROOT result = %s select(neg, neg_val, pos_sel)
`, shapeLiteral, moduleBuilder.elementType, lambda, shapeLiteral, shapeLiteral, shapeLiteral,
		shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardTanhRange(minVal, maxVal float64) string {
	body := moduleBuilder.parameter("p0") + moduleBuilder.clamp(minVal, maxVal, "p0", "result")
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSnakeParametric(alpha, beta float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  scaled = %s multiply(p0, alpha_b)
  sine = %s sine(scaled)
  sq = %s multiply(sine, sine)
  inv_beta = %s[] constant(%g)
  inv_beta_b = %s broadcast(inv_beta), dimensions={}
  addend = %s multiply(inv_beta_b, sq)
  ROOT result = %s add(p0, addend)
`, shapeLiteral, moduleBuilder.elementType, alpha, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, 1.0/beta, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderPReLUV() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pred = %s compare(p0, zero_b), direction=GT
  scaled = %s multiply(p0, p1)
  ROOT result = %s select(pred, p0, scaled)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderBinaryModule(body)
}

func (moduleBuilder *ModuleBuilder) renderRReLU(lower, upper float32) string {
	seed := uint32(0xA5A5A5A5) ^ math.Float32bits(lower) ^ math.Float32bits(upper)
	elementCount := moduleBuilder.elementCount()
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()
	lowerLiteral := fmt.Sprintf("%g", lower)
	spanLiteral := fmt.Sprintf("%g", float64(upper-lower))

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%lcg {
  state = u32[] parameter(0)
  mul = u32[] multiply(u32[] constant(1664525), state)
  ROOT next = u32[] add(mul, u32[] constant(1013904223))
}

%%update {
  state = u32[] parameter(0)
  elem = %s parameter(1)
  zero = %s[] constant(0)
  pred = pred[] compare(elem, zero), direction=LE
  advanced = u32[] call(state), to_apply=%%lcg
  new_state = u32[] select(pred, advanced, state)
  ROOT out = (u32[], u32[]) tuple(new_state, new_state)
}

ENTRY main {
  p0 = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pos = pred[] compare(p0, zero_b), direction=GT
  seed = u32[] constant(%d)
  states = u32[%d] scan(seed, p0), to_apply=%%update
  seed_vec = u32[1] reshape(seed), dimensions={0}
  prior_tail = u32[%d] slice(states), slice={[0]}, slice={[%d]}
  prev = u32[%d] concatenate(seed_vec, prior_tail), dimensions={0}
  advanced = u32[%d] map(prev), to_apply=%%lcg
  shift = u32[%d] shift-right-logical(advanced, u32[] constant(8))
  shift_f = %s convert(shift)
  norm = %s divide(shift_f, %s broadcast(%s[] constant(16777215), dimensions={}))
  slope = %s add(%s broadcast(%s[] constant(%s), dimensions={}), %s multiply(%s broadcast(%s[] constant(%s), dimensions={}), norm))
  neg_out = %s multiply(p0, slope)
  ROOT result = %s select(pos, p0, neg_out)
}
`, moduleBuilder.moduleName, entryLayout,
		shapeLiteral, moduleBuilder.elementType,
		shapeLiteral, moduleBuilder.elementType, shapeLiteral, seed,
		elementCount, elementCount-1, elementCount-1, elementCount, elementCount, elementCount,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, moduleBuilder.elementType,
		shapeLiteral, moduleBuilder.elementType, lowerLiteral, shapeLiteral, shapeLiteral, moduleBuilder.elementType, spanLiteral, shapeLiteral,
		shapeLiteral, shapeLiteral)
}

func (moduleBuilder *ModuleBuilder) elementCount() int64 {
	if len(moduleBuilder.dimensions) == 0 {
		return 1
	}

	total := int64(1)

	for _, dimension := range moduleBuilder.dimensions {
		total *= dimension
	}

	return total
}

func (moduleBuilder *ModuleBuilder) renderBinaryModule(body string) string {
	entryLayout := moduleBuilder.entryLayout()
	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s, %s}

ENTRY main {
%s
}
`, moduleBuilder.moduleName, entryLayout, entryLayout, body)
}
