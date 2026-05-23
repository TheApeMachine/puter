package dropout

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
    notImplemented(string)
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
