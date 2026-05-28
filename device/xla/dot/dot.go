package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Product implements device.Dot for the XLA backend.
*/
type Product struct {
	host Host
}

/*
Host is the XLA dispatch surface dot operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	DotProduct(
		dst unsafe.Pointer,
		left, right unsafe.Pointer,
		count int,
		format dtype.DType,
	)
}

/*
New wires a Product receiver to its XLA dispatch host.
*/
func New(host Host) Product {
	return Product{host: host}
}

func (product *Product) stubHost() {
	product.host.NeedsPlatform()
}

func (product *Product) unimplemented(methodName string) {
	product.host.NotImplemented(methodName)
}
