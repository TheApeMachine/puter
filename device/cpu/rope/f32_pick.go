//go:build amd64

package rope

type f32RopePairsKernelImpl struct {
	kernel    func(out, in, cosBuf, sinBuf []float32)
	name      string
	available bool
}

func pickF32RopePairsKernel(
	candidates []f32RopePairsKernelImpl,
) func(out, in, cosBuf, sinBuf []float32) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("rope: no float32 rope-pairs kernel available")
}
