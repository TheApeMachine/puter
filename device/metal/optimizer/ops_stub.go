//go:build !darwin || !cgo

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
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) Adam(
	config cpuoptimizer.AdamConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) Adamax(
	config cpuoptimizer.AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) AdamW(
	config cpuoptimizer.AdamWConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) Hebbian(
	config cpuoptimizer.HebbianConfig,
	weights, post, pre, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) LARS(
	config cpuoptimizer.LARSConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) LBFGS(
	config cpuoptimizer.LBFGSConfig,
	params, gradients, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) Lion(
	config cpuoptimizer.LionConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) RMSprop(
	config cpuoptimizer.RMSpropConfig,
	params, gradients, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}

func (optimizer Optimizer) SGD(
	config cpuoptimizer.SGDConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	optimizer.host.NeedsPlatform()
}
