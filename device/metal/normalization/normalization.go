package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

type Normalization struct {
	host Host
}

func New(host Host) Normalization {
	return Normalization{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchGroupNorm(
		config device.GroupNormConfig,
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	DispatchInstanceNorm(
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	DispatchBatchNormEval(
		input, scale, bias, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
}
