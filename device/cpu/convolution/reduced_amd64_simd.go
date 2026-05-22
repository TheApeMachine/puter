//go:build amd64

package convolution

import "golang.org/x/sys/cpu"

func reducedFloatSIMDAvailable() bool {
	return cpu.X86.HasAVX512F ||
		(cpu.X86.HasAVX2 && cpu.X86.HasFMA) ||
		cpu.X86.HasSSE2
}
