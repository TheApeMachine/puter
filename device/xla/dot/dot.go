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
    notImplemented(string)
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
