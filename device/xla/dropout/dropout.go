package dropout

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
DropoutLayer implements device.DropoutLayer for the XLA backend.
*/
type DropoutLayer struct {
	host Host
}

/*
Host is the XLA dispatch surface dropout operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchDropout(
		dst, src unsafe.Pointer,
		count int,
		config device.DropoutConfig,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a DropoutLayer receiver to its XLA dispatch host.
*/
func New(host Host) DropoutLayer {
	return DropoutLayer{host: host}
}

func (receiver *DropoutLayer) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *DropoutLayer) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
