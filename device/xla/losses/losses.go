package losses

/*
Losses implements device.Losses for the XLA backend.
*/
type Losses struct {
    host Host
}

/*
Host is the XLA dispatch surface losses operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Losses receiver to its XLA dispatch host.
*/
func New(host Host) Losses {
    return Losses{host: host}
}

func (receiver *Losses) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Losses) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
