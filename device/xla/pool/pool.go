package pool

/*
Pool implements device.Pool for the XLA backend.
*/
type Pool struct {
    host Host
}

/*
Host is the XLA dispatch surface pool operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Pool receiver to its XLA dispatch host.
*/
func New(host Host) Pool {
    return Pool{host: host}
}

func (receiver *Pool) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Pool) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
