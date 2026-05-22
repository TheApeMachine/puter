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

var pageWriteF32Funcs = []f32PageWriteKernelImpl{
	{PageWriteFloat32NEON, "neon", true},
	{pageWriteF32Generic, "generic", true},
}

var pageGatherF32Funcs = []f32PageGatherKernelImpl{
	{PageGatherFloat32NEON, "neon", true},
	{pageGatherF32Generic, "generic", true},
}

var pageWriteU16Funcs = []u16PageWriteKernelImpl{
	{PageWriteUint16NEON, "neon", true},
	{pageWriteU16Generic, "generic", true},
}

var pageGatherU16Funcs = []u16PageGatherKernelImpl{
	{PageGatherUint16NEON, "neon", true},
	{pageGatherU16Generic, "generic", true},
}

func whereF32NEON(dst, positive, negative *float32, mask []byte, count int) {
	WhereF32NEON(dst, positive, negative, mask, count)
}

func maskedFillF32NEON(dst, input *float32, fill float32, mask []byte, count int) {
	MaskedFillF32NEON(dst, input, fill, mask, count)
}
