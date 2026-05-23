package embedding

/*
Embedding implements device.Embedding for the XLA backend.
*/
type Embedding struct {
    host Host
}

/*
Host is the XLA dispatch surface embedding operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Embedding receiver to its XLA dispatch host.
*/
func New(host Host) Embedding {
    return Embedding{host: host}
}

func (receiver *Embedding) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Embedding) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
