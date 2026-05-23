package reduction

/*
Reduction implements device.Reduction for the XLA backend.
*/
type Reduction struct {
    host Host
}

/*
Host is the XLA dispatch surface reduction operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Reduction receiver to its XLA dispatch host.
*/
func New(host Host) Reduction {
    return Reduction{host: host}
}

func (receiver *Reduction) stubHost() {
    receiver.host.NeedsPlatform()
}
