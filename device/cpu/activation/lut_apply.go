package activation

import "unsafe"

func applyF16LUTScalar(dst, src *uint16, count int, lut *[65536]uint16) {
	in := unsafe.Slice(src, count)
	out := unsafe.Slice(dst, count)

	for index := range in {
		out[index] = lut[in[index]]
	}
}

func applyF16LUT(dst, src unsafe.Pointer, count int, lut *[65536]uint16) {
	if count == 0 {
		return
	}

	f16LUTGatherKernel(
		(*uint16)(dst),
		(*uint16)(src),
		count,
		lut,
	)
}

func applyBF16LUT(dst, src unsafe.Pointer, count int, lut *[65536]uint16) {
	applyF16LUT(dst, src, count, lut)
}
