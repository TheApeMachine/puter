package hlo

import "fmt"

func (moduleBuilder *ModuleBuilder) renderSilu() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  neg = %s negate(p0)
  exp = %s exponential(neg)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  denom = %s add(one_b, exp)
  ROOT result = %s divide(p0, denom)
`, shapeLiteral, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderGeluErf() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  inv_sqrt2 = %s[] constant(%g)
  inv_sqrt2_b = %s broadcast(inv_sqrt2), dimensions={}
  scaled = %s multiply(p0, inv_sqrt2_b)
  erf_val = %s erf(scaled)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, erf_val)
  half = %s[] constant(0.5)
  half_b = %s broadcast(half), dimensions={}
  scaled_inner = %s multiply(half_b, inner)
  ROOT result = %s multiply(p0, scaled_inner)
`, shapeLiteral, moduleBuilder.elementType, sqrtTwoInverse, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderGeluTanh() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  cube = %s multiply(p0, multiply(p0, p0))
  beta = %s[] constant(%g)
  beta_b = %s broadcast(beta), dimensions={}
  beta_cube = %s multiply(beta_b, cube)
  inner_sum = %s add(p0, beta_cube)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  scaled = %s multiply(alpha_b, inner_sum)
  tanh_val = %s tanh(scaled)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, tanh_val)
  half = %s[] constant(0.5)
  half_b = %s broadcast(half), dimensions={}
  scaled_inner = %s multiply(half_b, inner)
  ROOT result = %s multiply(p0, scaled_inner)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, geluTanhBeta, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, geluTanhAlpha, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderELU(alpha float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pred = %s compare(p0, zero_b), direction=GE
  exp = %s exponential(p0)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  exp_m1 = %s subtract(exp, one_b)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  neg = %s multiply(alpha_b, exp_m1)
  ROOT result = %s select(pred, p0, neg)
`, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral, moduleBuilder.elementType, alpha, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderCELU(alpha float64) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pred = %s compare(p0, zero_b), direction=GE
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  scaled = %s divide(p0, alpha_b)
  exp = %s exponential(scaled)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  exp_m1 = %s subtract(exp, one_b)
  neg = %s multiply(alpha_b, exp_m1)
  ROOT result = %s select(pred, p0, neg)
`, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, alpha, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSELU() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pred = %s compare(p0, zero_b), direction=GE
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={}
  pos = %s multiply(scale_b, p0)
  exp = %s exponential(p0)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  exp_m1 = %s subtract(exp, one_b)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  neg = %s multiply(scale_b, multiply(alpha_b, exp_m1))
  ROOT result = %s select(pred, pos, neg)
`, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, seluScale, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral, moduleBuilder.elementType, seluAlpha, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSoftplus() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  exp = %s exponential(p0)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, exp)
  ROOT result = %s log(inner)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderMish() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  exp = %s exponential(p0)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, exp)
  softplus = %s log(inner)
  tanh_val = %s tanh(softplus)
  ROOT result = %s multiply(p0, tanh_val)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSoftsign() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  abs_val = %s abs(p0)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  denom = %s add(one_b, abs_val)
  ROOT result = %s divide(p0, denom)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardSigmoid() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={}
  offset = %s[] constant(%g)
  offset_b = %s broadcast(offset), dimensions={}
  shifted = %s add(offset_b, multiply(p0, scale_b))
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  clamped_hi = %s minimum(shifted, one_b)
  ROOT result = %s maximum(clamped_hi, zero_b)
`, shapeLiteral, moduleBuilder.elementType, hardSigmoidScale, shapeLiteral,
		moduleBuilder.elementType, hardSigmoidOffset, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardSwish() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  three = %s[] constant(%g)
  three_b = %s broadcast(three), dimensions={}
  shifted = %s add(p0, three_b)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={}
  offset = %s[] constant(%g)
  offset_b = %s broadcast(offset), dimensions={}
  hs_in = %s add(offset_b, multiply(shifted, scale_b))
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  clamped_hi = %s minimum(hs_in, one_b)
  hs = %s maximum(clamped_hi, zero_b)
  ROOT result = %s multiply(p0, hs)
`, shapeLiteral, moduleBuilder.elementType, hardSwishOffset, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, hardSigmoidScale, shapeLiteral, moduleBuilder.elementType, hardSigmoidOffset, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardTanh() string {
	body := moduleBuilder.parameter("p0") + moduleBuilder.clamp(hardTanhMin, hardTanhMax, "p0", "result")
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderHardGelu() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  three = %s[] constant(%g)
  three_b = %s broadcast(three), dimensions={}
  inner = %s add(p0, three_b)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  six = %s[] constant(%g)
  six_b = %s broadcast(six), dimensions={}
  clamped_hi = %s minimum(inner, six_b)
  clamped = %s maximum(clamped_hi, zero_b)
  inv_six = %s[] constant(%g)
  inv_six_b = %s broadcast(inv_six), dimensions={}
  gate = %s multiply(clamped, inv_six_b)
  ROOT result = %s multiply(p0, gate)
`, shapeLiteral, moduleBuilder.elementType, hardGeluOffset, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, hardGeluClampUpper, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, 1.0/hardGeluClampUpper, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderQuickGelu() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={}
  scaled = %s multiply(scale_b, p0)
  sigmoid = %s logistic(scaled)
  ROOT result = %s multiply(p0, sigmoid)
`, shapeLiteral, moduleBuilder.elementType, quickGeluScale, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}

func (moduleBuilder *ModuleBuilder) renderTanhShrink() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  tanh_val = %s tanh(p0)
  ROOT result = %s subtract(p0, tanh_val)
`, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderModule(body)
}
