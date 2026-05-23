package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
DefaultBuilderTarget is the compile target string for GPU XLA programs.
*/
const DefaultBuilderTarget = "gpu"

/*
LoweringContext carries resident tensor metadata for an XLA lowering pass.
*/
type LoweringContext struct {
	Target      string
	InputDTypes []dtype.DType
	InputShapes []tensor.Shape
	OutputDType dtype.DType
	OutputShape tensor.Shape
}

/*
Lowering defines the contract for lowering a named operation into an XLA program key.
Concrete lowerings compile under the xla build tag through Builder.
*/
type Lowering interface {
	Name() string
	ProgramKey(context LoweringContext, floatParams []float64, intParams []int64) (ProgramKey, error)
}

/*
LoweringRegistry maps operation names to lowerings.
*/
type LoweringRegistry struct {
	lowerings map[string]Lowering
}

/*
NewLoweringRegistry constructs an empty lowering registry.
*/
func NewLoweringRegistry() *LoweringRegistry {
	return &LoweringRegistry{lowerings: make(map[string]Lowering)}
}

/*
Register adds a lowering under its Name().
*/
func (loweringRegistry *LoweringRegistry) Register(lowering Lowering) {
	loweringRegistry.lowerings[lowering.Name()] = lowering
}

/*
Lookup returns the lowering for an operation name.
*/
func (loweringRegistry *LoweringRegistry) Lookup(operationName string) (Lowering, bool) {
	lowering, ok := loweringRegistry.lowerings[operationName]
	return lowering, ok
}

/*
Builder compiles and executes XLA programs. The xla build tag provides PJRT wiring.
*/
type Builder struct {
	cache    *ExecutableCache
	registry *LoweringRegistry
	target   string
}

/*
NewBuilder constructs an XLA builder bound to a compile cache and lowering registry.
*/
func NewBuilder(
	executableCache *ExecutableCache,
	loweringRegistry *LoweringRegistry,
	target string,
) *Builder {
	return &Builder{
		cache:    executableCache,
		registry: loweringRegistry,
		target:   target,
	}
}

/*
ProgramKeyFor resolves the compile key for a registered operation.
*/
func (builder *Builder) ProgramKeyFor(
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	lowering, ok := builder.registry.Lookup(operationName)

	if !ok {
		return ProgramKey{}, &loweringError{message: "unknown XLA lowering: " + operationName}
	}

	programKey, err := lowering.ProgramKey(context, floatParams, intParams)

	if err != nil {
		return ProgramKey{}, err
	}

	programKey.Target = builder.target
	return programKey, nil
}

/*
CachedExecutable returns a compile-cache hit or nil when absent.
*/
func (builder *Builder) CachedExecutable(programKey ProgramKey) (*CompiledExecutable, bool) {
	return builder.cache.Lookup(programKey)
}

/*
RecordExecutable stores a compiled executable in the builder cache.
*/
func (builder *Builder) RecordExecutable(
	programKey ProgramKey,
	executable *CompiledExecutable,
) {
	builder.cache.Store(programKey, executable)
}

/*
NewDefaultBuilder constructs a builder with standard activation and elementwise lowerings.
*/
func NewDefaultBuilder(target string) *Builder {
	registry := NewLoweringRegistry()
	RegisterStandardActivations(registry)
	RegisterParametricActivations(registry)
	RegisterGatedActivations(registry)
	RegisterElementwiseLowerings(registry)

	return NewBuilder(NewExecutableCache(), registry, target)
}
