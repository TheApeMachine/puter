package rope

var ropePairsF32Kernel = func() func(out, in, cosBuf, sinBuf []float32) {
	return pickF32RopePairsKernel(ropePairsF32Funcs)
}()
