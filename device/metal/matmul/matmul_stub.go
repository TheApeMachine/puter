//go:build !darwin || !cgo

package matmul

func (gemm *Gemm) stubHost() {
	gemm.host.NeedsPlatform()
}
