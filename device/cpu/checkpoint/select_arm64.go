//go:build arm64

package checkpoint

var encodeFloat32DataFuncs = []float32DataEncodeKernelImpl{
	{checkpointEncodeFloat32DataNEON, "neon", true},
	{encodeFloat32DataScalar, "generic", true},
}

var decodeFloat32DataFuncs = []float32DataDecodeKernelImpl{
	{checkpointDecodeFloat32DataNEON, "neon", true},
	{decodeFloat32DataScalar, "generic", true},
}

func checkpointEncodeFloat32DataNEON(dst []byte, src []float32) {
	CheckpointEncodeFloat32DataNEON(&dst[0], &src[0], len(src))
}

func checkpointDecodeFloat32DataNEON(dst []float32, src []byte) {
	CheckpointDecodeFloat32DataNEON(&dst[0], &src[0], len(dst))
}
