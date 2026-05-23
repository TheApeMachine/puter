package active_inference

/*
ActiveInference implements device.ActiveInference for the XLA backend.
*/
type ActiveInference struct {
    host Host
}

/*
Host is the XLA dispatch surface active_inference operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a ActiveInference receiver to its XLA dispatch host.
*/
func New(host Host) ActiveInference {
    return ActiveInference{host: host}
}

func (receiver *ActiveInference) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *ActiveInference) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
