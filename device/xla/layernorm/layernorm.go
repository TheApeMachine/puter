package layernorm

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
    notImplemented(string)
}

/*
New wires a Norm receiver to its XLA dispatch host.
*/
func New(host Host) Norm {
    return Norm{host: host}
}

func (receiver *Norm) stubHost() {
    receiver.host.NeedsPlatform()
}
