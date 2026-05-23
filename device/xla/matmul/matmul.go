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
    notImplemented(string)
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
