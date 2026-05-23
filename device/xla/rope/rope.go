package rope

/*
RotaryEmbedding implements device.RotaryEmbedding for the XLA backend.
*/
type RotaryEmbedding struct {
    host Host
}

/*
Host is the XLA dispatch surface rope operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a RotaryEmbedding receiver to its XLA dispatch host.
*/
func New(host Host) RotaryEmbedding {
    return RotaryEmbedding{host: host}
}

func (receiver *RotaryEmbedding) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *RotaryEmbedding) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
