package physics

/*
Physics implements device.Physics for the XLA backend.
*/
type Physics struct {
    host Host
}

/*
Host is the XLA dispatch surface physics operations call into.
*/
type Host interface {
    NeedsPlatform()
    notImplemented(string)
}

/*
New wires a Physics receiver to its XLA dispatch host.
*/
func New(host Host) Physics {
    return Physics{host: host}
}

func (receiver *Physics) stubHost() {
    receiver.host.NeedsPlatform()
}
