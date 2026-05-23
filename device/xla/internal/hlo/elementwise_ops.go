package hlo

import "fmt"

func (moduleBuilder *ModuleBuilder) renderAxpy(floatParams []float64) (string, error) {
	if len(floatParams) == 0 {
		return "", fmt.Errorf("axpy requires alpha parameter")
	}

	alpha := floatParams[0]
	shapeLiteral := moduleBuilder.shapeLiteral()
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  p1 = %s parameter(1)
  alpha = %s[] constant(%g)
  alpha_b = %s broadcast(alpha), dimensions={}
  scaled = %s multiply(alpha_b, p1)
  ROOT result = %s add(p0, scaled)
`, shapeLiteral, shapeLiteral, moduleBuilder.elementType, alpha, shapeLiteral, shapeLiteral, shapeLiteral)

	return moduleBuilder.renderBinaryModule(body), nil
}
