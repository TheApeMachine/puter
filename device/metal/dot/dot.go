package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Product implements device.Dot for the Metal backend.
*/
type Product struct {
	host Host
}

/*
New wires a Product receiver to its Metal dispatch host.
*/
func New(host Host) Product {
	return Product{host: host}
}

/*
Host is the Metal dispatch surface dot operations call into.
*/
type Host interface {
	NeedsPlatform()
	DotProduct(
		left, right unsafe.Pointer,
		count int,
		format dtype.DType,
	) float32
}
