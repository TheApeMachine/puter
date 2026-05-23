package activation

import "github.com/theapemachine/puter/device/cpu/peel"

type lutGatherImpl struct {
	kernel    func(dst, src *uint16, count int, lut *[65536]uint16)
	name      string
	available bool
}

func pickLUTGatherKernel(
	candidates []lutGatherImpl,
) func(dst, src *uint16, count int, lut *[65536]uint16) {
	var genericKernel func(dst, src *uint16, count int, lut *[65536]uint16)

	for _, candidate := range candidates {
		if candidate.name == "generic" {
			genericKernel = candidate.kernel
		}
	}

	for _, candidate := range candidates {
		if !candidate.available {
			continue
		}

		if candidate.name == "generic" {
			return candidate.kernel
		}

		simdKernel := candidate.kernel

		return func(
			destination, source *uint16,
			count int,
			lut *[65536]uint16,
		) {
			peel.WrapF16Unary(
				func(dstLane, srcLane *uint16, laneCount int) {
					simdKernel(dstLane, srcLane, laneCount, lut)
				},
				func(dstLane, srcLane *uint16, laneCount int) {
					genericKernel(dstLane, srcLane, laneCount, lut)
				},
				candidate.name,
			)(destination, source, count)
		}
	}

	panic("activation: no f16/bf16 LUT gather kernel available")
}
