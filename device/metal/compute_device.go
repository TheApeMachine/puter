package metal

import (
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/pospop"
)

/*
ComputeDevice implements device.Backend by dispatching to Metal graph kernels
and host-side PosPop utilities. Graph execution uses DispatchGraph on tensors
already resident on the Metal memory backend.
*/
type ComputeDevice struct {
	memory *Backend
}

/*
NewComputeDevice constructs a device.Backend backed by Metal compute kernels.
*/
func NewComputeDevice(memory *Backend) *ComputeDevice {
	return &ComputeDevice{memory: memory}
}

/*
DispatchGraph runs one named Metal kernel against resident tensors.
*/
func (computeDevice *ComputeDevice) DispatchGraph(name string, args ...tensor.Tensor) error {
	return DispatchGraphKernel(name, args...)
}

func (computeDevice *ComputeDevice) CountString(counts *[8]int, str string) {
	pospop.CountString(counts, str)
}

func (computeDevice *ComputeDevice) Count8(counts *[8]int, buf []uint8) {
	pospop.Count8(counts, buf)
}

func (computeDevice *ComputeDevice) Count16(counts *[16]int, buf []uint16) {
	pospop.Count16(counts, buf)
}

func (computeDevice *ComputeDevice) Count32(counts *[32]int, buf []uint32) {
	pospop.Count32(counts, buf)
}

func (computeDevice *ComputeDevice) Count64(counts *[64]int, buf []uint64) {
	pospop.Count64(counts, buf)
}
