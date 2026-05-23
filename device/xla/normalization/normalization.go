package normalization

/*
Normalization implements device.Normalization for the XLA backend.
*/
type Normalization struct {
    host Host
}

/*
Host is the XLA dispatch surface normalization operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Normalization receiver to its XLA dispatch host.
*/
func New(host Host) Normalization {
    return Normalization{host: host}
}

func (receiver *Normalization) stubHost() {
    receiver.host.NeedsPlatform()
}
