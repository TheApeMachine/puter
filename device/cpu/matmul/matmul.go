package matmul

/*
Gemm implements device.{'Matmul'} for the CPU backend.
*/
type Gemm struct{}

/*
New constructs a Gemm receiver for CPU dispatch.
*/
func New() Gemm {
	return Gemm{}
}

var Default = New()
