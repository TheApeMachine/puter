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
	DispatchGreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
	DispatchTopKSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
	DispatchTopPSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
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
