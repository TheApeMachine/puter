package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
LossKernel selects a CUDA pair-loss kernel.
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
Losses implements device.Losses for the CUDA backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Losses struct {
	host Host
}

/*
New wires a Losses receiver to its CUDA dispatch host.
*/
func New(host Host) Losses {
	return Losses{host: host}
}

/*
Host is the CUDA dispatch surface loss operations call into.
*/
type Host interface {
	NeedsPlatform()
	PairLossScalar(
		predictions, targets unsafe.Pointer,
		format dtype.DType,
		kernel LossKernel,
	) float32
	CrossEntropyScalar(
		logits, targets unsafe.Pointer,
		batchSize, classes int,
		format dtype.DType,
	) float32
}
