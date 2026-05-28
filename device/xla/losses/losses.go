package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
LossKernel selects an XLA pair-loss program.
*/
type LossKernel int

const (
	KernelMSE LossKernel = iota
	KernelMAE
	KernelHuber
	KernelBinaryCrossEntropy
	KernelKLDivergence
)

/*
Losses implements device.Losses for the XLA backend.
*/
type Losses struct {
	host Host
}

/*
Host is the XLA dispatch surface losses operations call into.
*/
type Host interface {
	NeedsPlatform()
	PairLossScalar(
		dst unsafe.Pointer,
		predictions, targets unsafe.Pointer,
		count int,
		format dtype.DType,
		kernel LossKernel,
	)
	CrossEntropyScalar(
		dst unsafe.Pointer,
		logits, targets unsafe.Pointer,
		batchSize, classes int,
		format dtype.DType,
	)
}

/*
New wires a Losses receiver to its XLA dispatch host.
*/
func New(host Host) Losses {
	return Losses{host: host}
}

func (receiver *Losses) stubHost() {
	receiver.host.NeedsPlatform()
}
