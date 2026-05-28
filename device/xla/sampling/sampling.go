package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Sampling implements device.Sampling for the XLA backend.
*/
type Sampling struct {
	host Host
}

/*
Host is the XLA dispatch surface sampling operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchGreedySample(dst, logits unsafe.Pointer, vocabSize int, format dtype.DType)
	DispatchTopKSample(dst unsafe.Pointer, config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType)
	DispatchTopPSample(dst unsafe.Pointer, config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType)
	NotImplemented(string)
}

/*
New wires a Sampling receiver to its XLA dispatch host.
*/
func New(host Host) Sampling {
	return Sampling{host: host}
}

func (receiver *Sampling) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Sampling) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
