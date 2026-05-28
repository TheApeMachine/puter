package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

/*
valueTable maps ast.Graph value names (port names) to their runtime values
within a single CallGraph invocation. The table is seeded from the caller's
inputs and grows as each node writes its output(s).

Values can be:

  - tensor.Tensor — a resident device or host tensor (typical compute path)
  - []int / []float32 — small host-side payloads (token IDs, scalar
    sequences, schedule arrays produced by atomics like math.linspace)
  - int / float32 — single scalars (step indices, sigma deltas)

The table never owns the values: tensors are released by the caller when
the graph call returns, and host-side payloads are GC-managed.
*/
type valueTable struct {
	values     map[string]any
	slotByName map[string]int
	slots      []any
}

func newValueTable() *valueTable {
	return &valueTable{values: make(map[string]any)}
}

func newValueTableWithSlots(slotByName map[string]int, slotCount int) *valueTable {
	table := &valueTable{
		values:     make(map[string]any),
		slotByName: make(map[string]int, len(slotByName)),
		slots:      make([]any, slotCount),
	}

	for name, slot := range slotByName {
		table.slotByName[name] = slot
	}

	return table
}

func (table *valueTable) set(name string, value any) {
	if name == "" {
		return
	}

	if table.slotByName != nil {
		if slot, ok := table.slotByName[name]; ok {
			table.setSlot(slot, value)
			return
		}
	}

	table.values[name] = value
}

func (table *valueTable) get(name string) (any, bool) {
	if table.slotByName != nil {
		if slot, ok := table.slotByName[name]; ok {
			return table.getSlot(slot)
		}
	}

	value, ok := table.values[name]
	return value, ok
}

func (table *valueTable) setSlot(slot int, value any) {
	if !table.hasSlot(slot) {
		return
	}

	table.slots[slot] = value
}

func (table *valueTable) getSlot(slot int) (any, bool) {
	if !table.hasSlot(slot) {
		return nil, false
	}

	value := table.slots[slot]

	if value == nil {
		return nil, false
	}

	return value, true
}

func (table *valueTable) hasSlot(slot int) bool {
	return slot >= 0 && slot < len(table.slots)
}

func (table *valueTable) tensor(name string) (tensor.Tensor, error) {
	raw, ok := table.get(name)

	if !ok {
		return nil, fmt.Errorf("execution: value %q not found", name)
	}

	tensorValue, ok := raw.(tensor.Tensor)

	if !ok {
		return nil, fmt.Errorf("execution: value %q has type %T, expected tensor.Tensor", name, raw)
	}

	return tensorValue, nil
}

func (table *valueTable) tokenIDs(name string) ([]int, error) {
	raw, ok := table.get(name)

	if !ok {
		return nil, fmt.Errorf("execution: value %q not found", name)
	}

	switch typed := raw.(type) {
	case []int:
		return typed, nil
	case []int32:
		out := make([]int, len(typed))

		for index, value := range typed {
			out[index] = int(value)
		}

		return out, nil
	case []int64:
		out := make([]int, len(typed))

		for index, value := range typed {
			out[index] = int(value)
		}

		return out, nil
	case int:
		return []int{typed}, nil
	case int32:
		return []int{int(typed)}, nil
	case int64:
		return []int{int(typed)}, nil
	default:
		return nil, fmt.Errorf("execution: value %q has type %T, expected []int", name, raw)
	}
}
