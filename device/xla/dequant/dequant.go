package dequant

/*
Dequantization implements device.Dequantization for the XLA backend.
*/
type Dequantization struct {
    host Host
}

/*
Host is the XLA dispatch surface dequant operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Dequantization receiver to its XLA dispatch host.
*/
func New(host Host) Dequantization {
    return Dequantization{host: host}
}

func (receiver *Dequantization) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Dequantization) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
