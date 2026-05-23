package masking

/*
Masking implements device.Masking for the XLA backend.
*/
type Masking struct {
    host Host
}

/*
Host is the XLA dispatch surface masking operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Masking receiver to its XLA dispatch host.
*/
func New(host Host) Masking {
    return Masking{host: host}
}

func (receiver *Masking) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Masking) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
