package hlo

import "fmt"

func (moduleBuilder *ModuleBuilder) renderModule(body string) string {
	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
%s
}
`, moduleBuilder.moduleName, moduleBuilder.entryLayout(), body)
}

func (moduleBuilder *ModuleBuilder) broadcastScalar(scalarName, valueLiteral string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	return fmt.Sprintf(`  %s = %s[] constant(%s)
  %s = %s broadcast(%s), dimensions={}
`, scalarName, moduleBuilder.elementType, valueLiteral, scalarName+"_b", shapeLiteral, scalarName)
}

func (moduleBuilder *ModuleBuilder) parameter(name string) string {
	return fmt.Sprintf("  %s = %s parameter(0)", name, moduleBuilder.shapeLiteral())
}

func (moduleBuilder *ModuleBuilder) clamp(minVal, maxVal float64, valueName, resultName string) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	return fmt.Sprintf(`  %s_min = %s[] constant(%g)
  %s_min_b = %s broadcast(%s_min), dimensions={}
  %s_max = %s[] constant(%g)
  %s_max_b = %s broadcast(%s_max), dimensions={}
  %s_clamped_hi = %s minimum(%s, %s_max_b)
  ROOT %s = %s maximum(%s_clamped_hi, %s_min_b)
`, resultName, moduleBuilder.elementType, minVal, resultName, shapeLiteral, resultName,
		resultName, moduleBuilder.elementType, maxVal, resultName, shapeLiteral, resultName,
		resultName, shapeLiteral, valueName, resultName, shapeLiteral, resultName, resultName, resultName)
}

func (moduleBuilder *ModuleBuilder) compareSelectGE(
	valueName string,
	posExpr string,
	negExpr string,
	resultName string,
) string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	return fmt.Sprintf(`  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  pred = %s compare(%s, zero_b), direction=GE
  pos = %s %s
  neg = %s %s
  ROOT %s = %s select(pred, pos, neg)
`, moduleBuilder.elementType, shapeLiteral, shapeLiteral, valueName, shapeLiteral, posExpr, shapeLiteral, negExpr, resultName, shapeLiteral)
}
