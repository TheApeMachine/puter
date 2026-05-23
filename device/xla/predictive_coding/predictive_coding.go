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
    notImplemented(string)
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
