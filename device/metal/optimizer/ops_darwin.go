//go:build darwin && cgo

package optimizer

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"
)

func (optimizer Optimizer) Adagrad(
	config cpuoptimizer.AdagradConfig,
	params, gradients, accumulator, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchAdagrad(config, params, gradients, accumulator, output, count, format)
}

func (optimizer Optimizer) Adam(
	config cpuoptimizer.AdamConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchAdam(config, params, gradients, firstMoment, secondMoment, output, count, format)
}

func (optimizer Optimizer) Adamax(
	config cpuoptimizer.AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchAdamax(config, params, gradients, firstMoment, infinityMoment, output, count, format)
}

func (optimizer Optimizer) AdamW(
	config cpuoptimizer.AdamWConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchAdamW(config, params, gradients, firstMoment, secondMoment, output, count, format)
}

func (optimizer Optimizer) Hebbian(
	config cpuoptimizer.HebbianConfig,
	weights, post, pre, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchHebbian(config, weights, post, pre, output, count, format)
}

func (optimizer Optimizer) LARS(
	config cpuoptimizer.LARSConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchLARS(config, params, gradients, momentum, output, count, format)
}

func (optimizer Optimizer) LBFGS(
	config cpuoptimizer.LBFGSConfig,
	params, gradients, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchLBFGS(config, params, gradients, output, count, format)
}

func (optimizer Optimizer) Lion(
	config cpuoptimizer.LionConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchLion(config, params, gradients, momentum, output, count, format)
}

func (optimizer Optimizer) RMSprop(
	config cpuoptimizer.RMSpropConfig,
	params, gradients, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchRMSprop(config, params, gradients, secondMoment, output, count, format)
}

func (optimizer Optimizer) SGD(
	config cpuoptimizer.SGDConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.DispatchSGD(config, params, gradients, momentum, output, count, format)
}
