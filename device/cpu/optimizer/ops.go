package optimizer

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func requireOptimizerFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("optimizer: unsupported dtype")
	}
}

func float32SliceAt(pointer unsafe.Pointer, count int) []float32 {
	return dispatch.Float32Slice(pointer, count)
}

func hebbianMatrixDim(weightCount int) int {
	dimension := int(math.Sqrt(float64(weightCount)))

	if dimension*dimension != weightCount {
		panic("optimizer: hebbian weight count must be a perfect square")
	}

	return dimension
}

func (stepper Stepper) Adagrad(
	config AdagradConfig,
	params, gradients, accumulator, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	adagradStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(accumulator, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) Adam(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	adamStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(firstMoment, count),
		float32SliceAt(secondMoment, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) Adamax(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	adamaxStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(firstMoment, count),
		float32SliceAt(infinityMoment, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) AdamW(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	adamWStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(firstMoment, count),
		float32SliceAt(secondMoment, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) Hebbian(
	config HebbianConfig,
	weights, post, pre, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	dimension := hebbianMatrixDim(count)

	hebbianStepSlices(
		config,
		float32SliceAt(weights, count),
		float32SliceAt(post, dimension),
		float32SliceAt(pre, dimension),
		float32SliceAt(output, count),
		dimension,
	)
}

func (stepper Stepper) LARS(
	config LARSConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	larsStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(momentum, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) LBFGS(
	config LBFGSConfig,
	params, gradients, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	lbfgsStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) Lion(
	config LionConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	lionStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(momentum, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) RMSprop(
	config RMSpropConfig,
	params, gradients, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	rmspropStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(secondMoment, count),
		float32SliceAt(output, count),
	)
}

func (stepper Stepper) SGD(
	config SGDConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireOptimizerFloat32(format)

	if count == 0 {
		return
	}

	sgdStepSlices(
		config,
		float32SliceAt(params, count),
		float32SliceAt(gradients, count),
		float32SliceAt(momentum, count),
		float32SliceAt(output, count),
	)
}
