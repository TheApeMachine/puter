package activation

var f16LUTGatherKernel = func() func(dst, src *uint16, count int, lut *[65536]uint16) {
	return pickLUTGatherKernel(f16LUTGatherFuncs)
}()
