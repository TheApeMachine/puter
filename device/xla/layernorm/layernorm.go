package layernorm

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Norm implements device.Norm for the XLA backend.
*/
type Norm struct {
	host Host
}

/*
Host is the XLA dispatch surface layernorm operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	LaunchLayerNorm(
		input, scale, bias, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
	LaunchRMSNorm(
		config device.RMSNormConfig,
		input, scale, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
	LaunchModulatedLayerNorm(
		config device.ModulatedLayerNormConfig,
		input, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols int,
		format dtype.DType,
	)
}

/*
New wires a Norm receiver to its XLA dispatch host.
*/
func New(host Host) Norm {
	return Norm{host: host}
}

func (norm *Norm) stubHost() {
	norm.host.NeedsPlatform()
}

func (norm *Norm) unimplemented(methodName string) {
	norm.host.NotImplemented(methodName)
}
