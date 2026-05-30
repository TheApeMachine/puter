package geometry

import "unsafe"

var (
	phaseCouplingFloat32Kernel = func() func(
		destination, leftGrowth, rightGrowth []float32,
		count int,
	) {
		return pickPhaseCouplingFloat32Kernel(phaseCouplingFloat32Funcs)
	}()

	phaseCouplingFloat16Kernel = func() func(
		destination, leftGrowth, rightGrowth []uint16,
		count int,
	) {
		return pickPhaseCouplingUInt16Kernel(phaseCouplingFloat16Funcs)
	}()

	phaseCouplingBFloat16Kernel = func() func(
		destination, leftGrowth, rightGrowth []uint16,
		count int,
	) {
		return pickPhaseCouplingUInt16Kernel(phaseCouplingBFloat16Funcs)
	}()
)

func runPhaseCouplingFloat32(
	destination, leftGrowth, rightGrowth unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	phaseCouplingFloat32Kernel(
		unsafe.Slice((*float32)(destination), count),
		unsafe.Slice((*float32)(leftGrowth), count),
		unsafe.Slice((*float32)(rightGrowth), count),
		count,
	)
}

func runPhaseCouplingFloat16(
	destination, leftGrowth, rightGrowth unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	phaseCouplingFloat16Kernel(
		unsafe.Slice((*uint16)(destination), count),
		unsafe.Slice((*uint16)(leftGrowth), count),
		unsafe.Slice((*uint16)(rightGrowth), count),
		count,
	)
}

func runPhaseCouplingBFloat16(
	destination, leftGrowth, rightGrowth unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	phaseCouplingBFloat16Kernel(
		unsafe.Slice((*uint16)(destination), count),
		unsafe.Slice((*uint16)(leftGrowth), count),
		unsafe.Slice((*uint16)(rightGrowth), count),
		count,
	)
}
