package sampling

/*
Sampling implements device.Sampling for the XLA backend.
*/
type Sampling struct {
    host Host
}

/*
Host is the XLA dispatch surface sampling operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Sampling receiver to its XLA dispatch host.
*/
func New(host Host) Sampling {
    return Sampling{host: host}
}

func (receiver *Sampling) stubHost() {
    receiver.host.NeedsPlatform()
}
