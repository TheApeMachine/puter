package shape

var copyContiguousF32Kernel = func() func(dst, src *float32, count int) {
	return pickF32CopyContiguousKernel(copyContiguousF32Funcs)
}()

var whereF32Kernel = func() func(dst, positive, negative *float32, mask []byte, count int) {
	return pickF32WhereKernel(whereF32Funcs)
}()

var maskedFillF32Kernel = func() func(dst, input *float32, fill float32, mask []byte, count int) {
	return pickF32MaskedFillKernel(maskedFillF32Funcs)
}()

var pageWriteF32Kernel = func() f32PageWriteKernelImpl {
	return pickF32PageWriteKernel(pageWriteF32Funcs)
}()

var pageGatherF32Kernel = func() f32PageGatherKernelImpl {
	return pickF32PageGatherKernel(pageGatherF32Funcs)
}()

var pageWriteU16Kernel = func() u16PageWriteKernelImpl {
	return pickU16PageWriteKernel(pageWriteU16Funcs)
}()

var pageGatherU16Kernel = func() u16PageGatherKernelImpl {
	return pickU16PageGatherKernel(pageGatherU16Funcs)
}()
