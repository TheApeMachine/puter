package vsa

/*
VSA implements device.VSA for the XLA backend.
*/
type VSA struct {
    host Host
}

/*
Host is the XLA dispatch surface vsa operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a VSA receiver to its XLA dispatch host.
*/
func New(host Host) VSA {
    return VSA{host: host}
}

func (receiver *VSA) stubHost() {
    receiver.host.NeedsPlatform()
}
