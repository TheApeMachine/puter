package optimizer

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

// Kernel aliases kernels.Kernel for consolidated registry access.
type Kernel = kernels.Kernel

// Signature aliases kernels.Signature.
type Signature = kernels.Signature

// Default re-exports kernels.Default as the package default registry.
var Default = kernels.Default

/*
Mixed-precision optimizer dispatchers. Standard AMP/bf16 training
convention: params, gradients, and output stored at the reduced dtype
(bf16 or fp16); all optimizer state (momentum, variance, etc.) stays
at f32 for numerical stability across the long-tail of training steps.

Each step reads params and gradients lane-wise at native width, updates
f32 state in place, and stores the output at native width.
*/

type mixedOptimizerStep func(
	count int,
	loadParam, loadGrad valueLoad,
	state [][]float32,
	storeOut valueStore,
)

func runMixedOptimizerBFloat16(args []tensor.Tensor, stateCount int, step mixedOptimizerStep) error {
	if len(args) != stateCount+3 {
		return tensor.ErrShapeMismatch
	}

	paramsBF16, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	gradBF16, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	outBF16, err := args[stateCount+2].BFloat16Native()

	if err != nil {
		return err
	}

	count := len(paramsBF16)

	if len(gradBF16) != count || len(outBF16) != count {
		return tensor.ErrShapeMismatch
	}

	stateSlices := make([][]float32, stateCount)

	for index := range stateCount {
		state, err := args[2+index].Float32Native()

		if err != nil {
			return err
		}

		if len(state) != count {
			return tensor.ErrShapeMismatch
		}

		stateSlices[index] = state
	}

	step(
		count,
		func(index int) float32 { return (&paramsBF16[index]).Float32() },
		func(index int) float32 { return (&gradBF16[index]).Float32() },
		stateSlices,
		func(index int, value float32) {
			outBF16[index] = dtype.NewBfloat16FromFloat32(value)
		},
	)

	return nil
}

func runMixedOptimizerFloat16(args []tensor.Tensor, stateCount int, step mixedOptimizerStep) error {
	if len(args) != stateCount+3 {
		return tensor.ErrShapeMismatch
	}

	paramsF16, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	gradF16, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	outF16, err := args[stateCount+2].Float16Native()

	if err != nil {
		return err
	}

	count := len(paramsF16)

	if len(gradF16) != count || len(outF16) != count {
		return tensor.ErrShapeMismatch
	}

	stateSlices := make([][]float32, stateCount)

	for index := range stateCount {
		state, err := args[2+index].Float32Native()

		if err != nil {
			return err
		}

		if len(state) != count {
			return tensor.ErrShapeMismatch
		}

		stateSlices[index] = state
	}

	step(
		count,
		func(index int) float32 { return (&paramsF16[index]).Float32() },
		func(index int) float32 { return (&gradF16[index]).Float32() },
		stateSlices,
		func(index int, value float32) {
			outF16[index] = dtype.Fromfloat32(value)
		},
	)

	return nil
}

func registerMixedOptimizer(name string, stateCount int, step mixedOptimizerStep) {
	inputDtypesBF16 := make([]dtype.DType, 2+stateCount)
	inputDtypesBF16[0] = dtype.BFloat16
	inputDtypesBF16[1] = dtype.BFloat16

	inputDtypesF16 := make([]dtype.DType, 2+stateCount)
	inputDtypesF16[0] = dtype.Float16
	inputDtypesF16[1] = dtype.Float16

	for index := range stateCount {
		inputDtypesBF16[2+index] = dtype.Float32
		inputDtypesF16[2+index] = dtype.Float32
	}

	bfRun := func(args ...tensor.Tensor) error {
		return runMixedOptimizerBFloat16(args, stateCount, step)
	}

	f16Run := func(args ...tensor.Tensor) error {
		return runMixedOptimizerFloat16(args, stateCount, step)
	}

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDtypesBF16,
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       bfRun,
	})

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDtypesF16,
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       f16Run,
	})
}

// RegisterMixedPrecisionSteps registers bf16/fp16 optimizer kernels whose
// state tensors remain f32. Called from the neon CPU kernel registry init.
func RegisterMixedPrecisionSteps() {
	registerMixedOptimizer("adam_step", 2, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		adamMixedStep(DefaultAdamConfig(), count, loadParam, loadGrad, state[0], state[1], storeOut)
	})

	registerMixedOptimizer("adamw_step", 2, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		adamWMixedStep(DefaultAdamWConfig(), count, loadParam, loadGrad, state[0], state[1], storeOut)
	})

	registerMixedOptimizer("lion_step", 1, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		lionMixedStep(DefaultLionConfig(), count, loadParam, loadGrad, state[0], storeOut)
	})

	registerMixedOptimizer("sgd_step", 1, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		sgdMixedStep(DefaultSGDConfig(), count, loadParam, loadGrad, state[0], storeOut)
	})

	registerMixedOptimizer("adamax_step", 2, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		adamaxMixedStep(DefaultAdamaxConfig(), count, loadParam, loadGrad, state[0], state[1], storeOut)
	})

	registerMixedOptimizer("adagrad_step", 1, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		adagradMixedStep(DefaultAdagradConfig(), count, loadParam, loadGrad, state[0], storeOut)
	})

	registerMixedOptimizer("rmsprop_step", 1, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		rmspropMixedStep(DefaultRMSpropConfig(), count, loadParam, loadGrad, state[0], storeOut)
	})

	registerMixedOptimizer("lars_step", 1, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		larsMixedStep(DefaultLARSConfig(), count, loadParam, loadGrad, state[0], storeOut)
	})

	registerMixedOptimizer("lbfgs_step", 0, func(count int, loadParam, loadGrad valueLoad, state [][]float32, storeOut valueStore) {
		lbfgsMixedStep(DefaultLBFGSConfig(), count, loadParam, loadGrad, storeOut)
	})

	registerHebbianMixedDType()
}

func registerHebbianMixedDType() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType
		Default.Register(Kernel{
			Name: "hebbian_step",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{paramDType, paramDType, paramDType},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				if paramDType == dtype.BFloat16 {
					return runHebbianBFloat16(args)
				}

				return runHebbianFloat16(args)
			},
		})
	}
}

func runHebbianBFloat16(args []tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	weightsBF16, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	postBF16, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	preBF16, err := args[2].BFloat16Native()

	if err != nil {
		return err
	}

	outBF16, err := args[3].BFloat16Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != len(postBF16) || dims[1] != len(preBF16) ||
		len(outBF16) != len(weightsBF16) {
		return tensor.ErrShapeMismatch
	}

	hebbianMixedStep(
		DefaultHebbianConfig(),
		func(index int) float32 { return (&weightsBF16[index]).Float32() },
		func(index int) float32 { return (&postBF16[index]).Float32() },
		func(index int) float32 { return (&preBF16[index]).Float32() },
		func(index int, value float32) {
			outBF16[index] = dtype.NewBfloat16FromFloat32(value)
		},
		len(postBF16), dims[1],
	)

	return nil
}

func runHebbianFloat16(args []tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	weightsF16, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	postF16, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	preF16, err := args[2].Float16Native()

	if err != nil {
		return err
	}

	outF16, err := args[3].Float16Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != len(postF16) || dims[1] != len(preF16) ||
		len(outF16) != len(weightsF16) {
		return tensor.ErrShapeMismatch
	}

	hebbianMixedStep(
		DefaultHebbianConfig(),
		func(index int) float32 { return (&weightsF16[index]).Float32() },
		func(index int) float32 { return (&postF16[index]).Float32() },
		func(index int) float32 { return (&preF16[index]).Float32() },
		func(index int, value float32) {
			outF16[index] = dtype.Fromfloat32(value)
		},
		len(postF16), dims[1],
	)

	return nil
}
