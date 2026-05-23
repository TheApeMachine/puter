package causal

/*
Causal implements device.Causal for the XLA backend.
*/
type Causal struct {
    host Host
}

/*
Host is the XLA dispatch surface causal operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Causal receiver to its XLA dispatch host.
*/
func New(host Host) Causal {
    return Causal{host: host}
}

func (receiver *Causal) stubHost() {
    receiver.host.NeedsPlatform()
}
