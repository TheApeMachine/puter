//go:build arm64

package tokenizer

var packInt32Funcs = []int32PackKernelImpl{
	{TokenizerPackInt32NEON, "neon", true},
	{packInt32Generic, "generic", true},
}
