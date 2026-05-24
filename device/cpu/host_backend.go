package cpu

import (
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/pospop"
)

/*
HostBackend implements device.HostBackend for CPU-side preprocessing.
PosPop never runs on the device execution path for Metal, CUDA, or XLA.
*/
type HostBackend struct {
	pospop.PosPop
}

/*
NewHostBackend constructs the CPU host preprocessing backend.
*/
func NewHostBackend() *HostBackend {
	hostBackend := &HostBackend{}
	hostBackend.PosPop = pospop.New()

	return hostBackend
}

var _ device.HostBackend = (*HostBackend)(nil)
