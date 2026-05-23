package dropout

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
DropoutLayer implements device.Dropout for the Metal backend.
*/
type DropoutLayer struct {
	host Host
}

func New(host Host) DropoutLayer {
	return DropoutLayer{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchDropout(
		dst, src unsafe.Pointer,
		count int,
		config device.DropoutConfig,
		format dtype.DType,
	)
}
