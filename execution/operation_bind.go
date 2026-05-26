package execution

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
)

/*
OperationBind is the executable mapping from one manifest operation to one
device.Backend method or executor intrinsic.
*/
type OperationBind struct {
	Operation     string
	InputNames    []string
	OutputNames   []string
	Method        string
	ConfigStruct  string
	ConfigFields  map[string]asset.BindArg
	Output        asset.BindOutput
	Args          []asset.BindArg
	selectedInput int
}

type operationSchema struct {
	schema asset.Schema
}

type operationRegistry struct {
	operations map[string]operationSchema
}

var (
	defaultRegistryOnce sync.Once
	defaultRegistry     *operationRegistry
	defaultRegistryErr  error
)

func defaultOperationRegistry() (*operationRegistry, error) {
	defaultRegistryOnce.Do(func() {
		defaultRegistry, defaultRegistryErr = loadOperationRegistry()
	})

	return defaultRegistry, defaultRegistryErr
}

func loadOperationRegistry() (*operationRegistry, error) {
	schemas, err := asset.Walk("template/operation")

	if err != nil {
		return nil, fmt.Errorf("execution: load operation assets: %w", err)
	}

	operations := make(map[string]operationSchema, len(schemas))

	for op, schema := range schemas {
		if schema.Bind == nil {
			continue
		}

		operations[op] = operationSchema{schema: schema}
	}

	return &operationRegistry{operations: operations}, nil
}

func (registry *operationRegistry) Bind(node *ast.GraphNode) (OperationBind, error) {
	if registry == nil {
		return OperationBind{}, fmt.Errorf("execution: operation registry is required")
	}

	if node == nil {
		return OperationBind{}, fmt.Errorf("execution: graph node is required")
	}

	entry, ok := registry.operations[node.Op]

	if !ok {
		return OperationBind{}, fmt.Errorf("unsupported op %q (no operation bind registered)", node.Op)
	}

	bind, err := entry.bindForNode(node)

	if err != nil {
		return OperationBind{}, err
	}

	bind.Operation = node.Op
	bind.InputNames = portNames(entry.schema.Inputs)
	bind.OutputNames = portNames(entry.schema.Outputs)

	return bind, nil
}

func (schema operationSchema) bindForNode(node *ast.GraphNode) (OperationBind, error) {
	raw := schema.schema.Bind

	for _, variant := range raw.Variants {
		if variant.When.InputCount == 0 {
			continue
		}

		if variant.When.InputCount != len(node.Inputs) {
			continue
		}

		return bindFromVariant(variant, len(node.Inputs)), nil
	}

	if len(raw.Variants) > 0 && raw.Method == "" {
		return OperationBind{}, fmt.Errorf(
			"operation %q has no bind variant for %d input(s)",
			node.Op, len(node.Inputs),
		)
	}

	return OperationBind{
		Method:        raw.Method,
		ConfigStruct:  raw.ConfigStruct,
		ConfigFields:  raw.ConfigFields,
		Output:        raw.Output,
		Args:          raw.Args,
		selectedInput: len(node.Inputs),
	}, nil
}

func bindFromVariant(variant asset.BindVariant, inputCount int) OperationBind {
	return OperationBind{
		Method:        variant.Method,
		ConfigStruct:  variant.ConfigStruct,
		ConfigFields:  variant.ConfigFields,
		Output:        variant.Output,
		Args:          variant.Args,
		selectedInput: inputCount,
	}
}

func portNames(ports []asset.OperationPort) []string {
	names := make([]string, len(ports))

	for index, port := range ports {
		names[index] = port.Name
	}

	return names
}
