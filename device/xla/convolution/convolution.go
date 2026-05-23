package convolution

/*
Convolution implements device.Convolution for the XLA backend.
*/
type Convolution struct {
    host Host
}

/*
Host is the XLA dispatch surface convolution operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Convolution receiver to its XLA dispatch host.
*/
func New(host Host) Convolution {
    return Convolution{host: host}
}

func (receiver *Convolution) stubHost() {
    receiver.host.NeedsPlatform()
}
