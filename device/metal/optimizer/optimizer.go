package optimizer

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"
)

/*
Optimizer implements device.Optimizer for the Metal backend.
*/
type Optimizer struct {
	host Host
}

/*
New wires an Optimizer receiver to its Metal dispatch host.
*/
func New(host Host) Optimizer {
	return Optimizer{host: host}
}

/*
Host is the Metal dispatch surface optimizer operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchAdagrad(
		config cpuoptimizer.AdagradConfig,
		params, gradients, accumulator, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchAdam(
		config cpuoptimizer.AdamConfig,
		params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchAdamax(
		config cpuoptimizer.AdamaxConfig,
		params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchAdamW(
		config cpuoptimizer.AdamWConfig,
		params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchHebbian(
		config cpuoptimizer.HebbianConfig,
		weights, post, pre, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchLARS(
		config cpuoptimizer.LARSConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchLBFGS(
		config cpuoptimizer.LBFGSConfig,
		params, gradients, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchLion(
		config cpuoptimizer.LionConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchRMSprop(
		config cpuoptimizer.RMSpropConfig,
		params, gradients, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchSGD(
		config cpuoptimizer.SGDConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
}
