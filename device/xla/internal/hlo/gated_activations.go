package hlo

import (
	"fmt"
)

func (moduleBuilder *ModuleBuilder) renderGatedBinary(operationName string) (string, error) {
	switch operationName {
	case "glu":
		return moduleBuilder.renderGLU("logistic"), nil
	case "geglu":
		return moduleBuilder.renderGLU("gelu_erf"), nil
	case "geglu_tanh":
		return moduleBuilder.renderGLU("gelu_tanh"), nil
	case "swiglu":
		return moduleBuilder.renderSwiGLU(), nil
	case "reglu":
		return moduleBuilder.renderGLU("relu"), nil
	case "siglu":
		return moduleBuilder.renderSiGLU(), nil
	case "linglu":
		return moduleBuilder.renderLinGLU(), nil
	case "seglu":
		return moduleBuilder.renderSeGLU(), nil
	default:
		return "", fmt.Errorf("unsupported gated HLO operation: %s", operationName)
	}
}

func (moduleBuilder *ModuleBuilder) renderGLU(activationKind string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	upBody := moduleBuilder.gateActivationBody("p1", activationKind)
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
%s  ROOT result = %s multiply(p0, activated)
`, shapeLiteral, shapeLiteral, upBody, shapeLiteral)
	return moduleBuilder.renderBinaryModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSwiGLU() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
  neg = %s negate(p0)
  exp = %s exponential(neg)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  denom = %s add(one_b, exp)
  silu = %s divide(p0, denom)
  ROOT result = %s multiply(silu, p1)
`, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderBinaryModule(body)
}

func (moduleBuilder *ModuleBuilder) renderLinGLU() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
  ROOT result = %s multiply(p0, p1)
`, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderBinaryModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSeGLU() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
  sig = %s logistic(p0)
  ROOT result = %s multiply(sig, p1)
`, shapeLiteral, shapeLiteral, shapeLiteral, shapeLiteral)
	return moduleBuilder.renderBinaryModule(body)
}

func (moduleBuilder *ModuleBuilder) renderSiGLU() string {
	return moduleBuilder.renderSeGLU()
}

func (moduleBuilder *ModuleBuilder) gateActivationBody(valueName, activationKind string) string {
	switch activationKind {
	case "logistic":
		return fmt.Sprintf("  activated = %s logistic(%s)\n", moduleBuilder.shapeLiteral(), valueName)
	case "relu":
		return fmt.Sprintf(`  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  activated = %s maximum(%s, zero_b)
`, moduleBuilder.elementType, moduleBuilder.shapeLiteral(), moduleBuilder.shapeLiteral(), valueName)
	case "gelu_erf":
		return moduleBuilder.geluErfBody(valueName, "activated")
	case "gelu_tanh":
		return moduleBuilder.geluTanhBody(valueName, "activated")
	default:
		return fmt.Sprintf("  activated = %s %s\n", moduleBuilder.shapeLiteral(), valueName)
	}
}

func (moduleBuilder *ModuleBuilder) geluErfBody(inputName, outputName string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	return fmt.Sprintf(`  inv_sqrt2 = %s[] constant(%g)
  inv_sqrt2_b = %s broadcast(inv_sqrt2), dimensions={}
  scaled = %s multiply(%s, inv_sqrt2_b)
  erf_val = %s erf(scaled)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, erf_val)
  half = %s[] constant(0.5)
  half_b = %s broadcast(half), dimensions={}
  scaled_inner = %s multiply(half_b, inner)
  %s = %s multiply(%s, scaled_inner)
`, moduleBuilder.elementType, sqrtTwoInverse, shapeLiteral, shapeLiteral, inputName, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, outputName, shapeLiteral, inputName)
}

func (moduleBuilder *ModuleBuilder) geluTanhBody(inputName, outputName string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	return fmt.Sprintf(`  cube = %s multiply(%s, multiply(%s, %s))
  beta = %s[] constant(%g)
  beta_b = %s broadcast(beta), dimensions={}
  beta_cube = %s multiply(beta_b, cube)
  inner_sum = %s add(%s, beta_cube)
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
  %s = %s multiply(%s, scaled_inner)
`, shapeLiteral, inputName, inputName, inputName, moduleBuilder.elementType, geluTanhBeta, shapeLiteral, shapeLiteral, shapeLiteral, inputName,
		moduleBuilder.elementType, geluTanhAlpha, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, shapeLiteral, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral, outputName, shapeLiteral, inputName)
}

func (moduleBuilder *ModuleBuilder) renderGLUPackedFromParams(operationName string, batch, halfCount int64) (string, error) {
	inType := fmt.Sprintf("%s[%d,%d]{1,0}", moduleBuilder.elementType, batch, halfCount*2)
	outType := fmt.Sprintf("%s[%d,%d]{1,0}", moduleBuilder.elementType, batch, halfCount)
	halfType := fmt.Sprintf("%s[%d,%d]{1,0}", moduleBuilder.elementType, batch, halfCount)

	sliceGate := fmt.Sprintf("  gate = %s slice(p0), slice={[0, 0]}, slice={[%d, %d]}\n", halfType, batch, halfCount)
	sliceUp := fmt.Sprintf("  up = %s slice(p0), slice={[0, %d]}, slice={[%d, %d]}\n", halfType, halfCount, batch, halfCount)

	var root string

	switch operationName {
	case "swiglu":
		root = fmt.Sprintf(`  neg = %s negate(gate)
  exp = %s exponential(neg)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  denom = %s add(one_b, exp)
  silu = %s divide(gate, denom)
  ROOT result = %s multiply(silu, up)
`, halfType, halfType, moduleBuilder.elementType, halfType, halfType, halfType, outType)
	case "siglu", "seglu":
		root = fmt.Sprintf(`  sig = %s logistic(gate)
  ROOT result = %s multiply(sig, up)
`, halfType, outType)
	case "linglu":
		root = fmt.Sprintf("  ROOT result = %s multiply(gate, up)\n", outType)
	case "geglu":
		root = moduleBuilder.geluErfBody("up", "activated") + fmt.Sprintf("  ROOT result = %s multiply(gate, activated)\n", outType)
	case "geglu_tanh":
		root = moduleBuilder.geluTanhBody("up", "activated") + fmt.Sprintf("  ROOT result = %s multiply(gate, activated)\n", outType)
	case "reglu":
		root = fmt.Sprintf(`  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  activated = %s maximum(up, zero_b)
  ROOT result = %s multiply(gate, activated)
`, moduleBuilder.elementType, halfType, halfType, outType)
	default:
		root = fmt.Sprintf(`  activated = %s logistic(up)
  ROOT result = %s multiply(gate, activated)
`, halfType, outType)
	}

	body := fmt.Sprintf("  p0 = %s parameter(0)\n%s%s%s", inType, sliceGate, sliceUp, root)
	entryLayout := fmt.Sprintf("%s->%s", inType, outType)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
%s
}
`, moduleBuilder.moduleName, entryLayout, body), nil
}
