//go:build amd64

package geometry

import "golang.org/x/sys/cpu"

//go:noescape
func hasAVX512FP16Asm() bool

var hasAVX512FP16 = func() bool {
	if !hasAVX512FP16Asm() {
		return false
	}

	return cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL
}()
