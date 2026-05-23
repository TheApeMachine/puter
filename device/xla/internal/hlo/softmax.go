package hlo

import "fmt"

func (moduleBuilder *ModuleBuilder) renderSoftmax() string {
	if len(moduleBuilder.dimensions) != 2 {
		return moduleBuilder.renderSoftmax1D()
	}

	rows := moduleBuilder.dimensions[0]
	cols := moduleBuilder.dimensions[1]
	rowType := fmt.Sprintf("%s[%d,%d]{1,0}", moduleBuilder.elementType, rows, cols)
	rowVecType := fmt.Sprintf("%s[%d]{0}", moduleBuilder.elementType, rows)
	entryLayout := fmt.Sprintf("%s->%s", rowType, rowType)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT m = %s[] maximum(lhs, rhs)
}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT sum = %s[] add(lhs, rhs)
}

ENTRY main {
  p0 = %s parameter(0)
  neg_inf = %s[] constant(-inf)
  row_max = %s reduce(p0, neg_inf), dimensions={1}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={0}
  shifted = %s subtract(p0, row_max_b)
  exp_val = %s exponential(shifted)
  zero = %s[] constant(0)
  row_sum = %s reduce(exp_val, zero), dimensions={1}, to_apply=%%add
  row_sum_b = %s broadcast(row_sum), dimensions={0}
  ROOT result = %s divide(exp_val, row_sum_b)
}
`, moduleBuilder.moduleName, entryLayout,
		moduleBuilder.elementType, moduleBuilder.elementType, moduleBuilder.elementType,
		moduleBuilder.elementType, moduleBuilder.elementType, moduleBuilder.elementType,
		rowType, moduleBuilder.elementType, rowVecType, rowVecType, rowType, rowType, rowType,
		rowVecType, rowVecType, rowType)
}

func (moduleBuilder *ModuleBuilder) renderSoftmax1D() string {
	shapeLiteral := moduleBuilder.shapeLiteral()
	entryLayout := moduleBuilder.entryLayout()

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT m = %s[] maximum(lhs, rhs)
}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT sum = %s[] add(lhs, rhs)
}

ENTRY main {
  p0 = %s parameter(0)
  neg_inf = %s[] constant(-inf)
  row_max = %s[] reduce(p0, neg_inf), dimensions={0}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={}
  shifted = %s subtract(p0, row_max_b)
  exp_val = %s exponential(shifted)
  zero = %s[] constant(0)
  row_sum = %s[] reduce(exp_val, zero), dimensions={0}, to_apply=%%add
  row_sum_b = %s broadcast(row_sum), dimensions={}
  ROOT result = %s divide(exp_val, row_sum_b)
}
`, moduleBuilder.moduleName, entryLayout,
		moduleBuilder.elementType, moduleBuilder.elementType, moduleBuilder.elementType,
		moduleBuilder.elementType, moduleBuilder.elementType, moduleBuilder.elementType,
		shapeLiteral, moduleBuilder.elementType, moduleBuilder.elementType, shapeLiteral, shapeLiteral, shapeLiteral,
		moduleBuilder.elementType, moduleBuilder.elementType, shapeLiteral, shapeLiteral)
}
