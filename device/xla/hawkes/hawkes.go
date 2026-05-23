package hawkes

/*
Hawkes implements device.Hawkes for the XLA backend.
*/
type Hawkes struct {
    host Host
}

/*
Host is the XLA dispatch surface hawkes operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Hawkes receiver to its XLA dispatch host.
*/
func New(host Host) Hawkes {
    return Hawkes{host: host}
}

func (receiver *Hawkes) stubHost() {
    receiver.host.NeedsPlatform()
}
