//go:build !cuda

package matmul

func (gemm *Gemm) stubHost() {
	gemm.host.NeedsPlatform()
}
