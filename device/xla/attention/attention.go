package attention

/*
Attention implements device.Attention for the XLA backend.
*/
type Attention struct {
    host Host
}

/*
Host is the XLA dispatch surface attention operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Attention receiver to its XLA dispatch host.
*/
func New(host Host) Attention {
    return Attention{host: host}
}

func (receiver *Attention) stubHost() {
    receiver.host.NeedsPlatform()
}
