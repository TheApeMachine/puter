package predictive_coding

/*
PredictiveCoding implements device.PredictiveCoding for the XLA backend.
*/
type PredictiveCoding struct {
    host Host
}

/*
Host is the XLA dispatch surface predictive_coding operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a PredictiveCoding receiver to its XLA dispatch host.
*/
func New(host Host) PredictiveCoding {
    return PredictiveCoding{host: host}
}

func (receiver *PredictiveCoding) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *PredictiveCoding) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
