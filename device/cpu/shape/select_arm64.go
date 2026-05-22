//go:build arm64

package shape

var copyContiguousF32Funcs = []f32CopyContiguousKernelImpl{
	{CopyContiguousF32NEON, "neon", true},
	{copyContiguousF32Generic, "generic", true},
}

var whereF32Funcs = []f32WhereKernelImpl{
	{whereF32NEON, "neon", true},
	{whereF32Generic, "generic", true},
}

var maskedFillF32Funcs = []f32MaskedFillKernelImpl{
	{maskedFillF32NEON, "neon", true},
	{maskedFillF32Generic, "generic", true},
}

func whereF32NEON(dst, positive, negative *float32, mask []byte, count int) {
	WhereF32NEON(dst, positive, negative, mask, count)
}

func maskedFillF32NEON(dst, input *float32, fill float32, mask []byte, count int) {
	MaskedFillF32NEON(dst, input, fill, mask, count)
}
