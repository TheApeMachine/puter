//go:build !amd64 && !arm64

package shape

var copyContiguousF32Funcs = []f32CopyContiguousKernelImpl{
	{copyContiguousF32Generic, "generic", true},
}

var whereF32Funcs = []f32WhereKernelImpl{
	{whereF32Generic, "generic", true},
}

var maskedFillF32Funcs = []f32MaskedFillKernelImpl{
	{maskedFillF32Generic, "generic", true},
}

var pageWriteF32Funcs = []f32PageWriteKernelImpl{
	{pageWriteF32Generic, "generic", true},
}

var pageGatherF32Funcs = []f32PageGatherKernelImpl{
	{pageGatherF32Generic, "generic", true},
}

var pageWriteU16Funcs = []u16PageWriteKernelImpl{
	{pageWriteU16Generic, "generic", true},
}

var pageGatherU16Funcs = []u16PageGatherKernelImpl{
	{pageGatherU16Generic, "generic", true},
}
