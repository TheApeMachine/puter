package dot

/*
Product implements device.Product for the XLA backend.
*/
type Product struct {
    host Host
}

/*
Host is the XLA dispatch surface dot operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Product receiver to its XLA dispatch host.
*/
func New(host Host) Product {
    return Product{host: host}
}

func (receiver *Product) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Product) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
