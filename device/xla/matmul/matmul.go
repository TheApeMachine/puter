package matmul

/*
Gemm implements device.Gemm for the XLA backend.
*/
type Gemm struct {
    host Host
}

/*
Host is the XLA dispatch surface matmul operations call into.
*/
type Host interface {
    NeedsPlatform()
    NotImplemented(string)
}

/*
New wires a Gemm receiver to its XLA dispatch host.
*/
func New(host Host) Gemm {
    return Gemm{host: host}
}

func (receiver *Gemm) stubHost() {
    receiver.host.NeedsPlatform()
}

func (receiver *Gemm) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
