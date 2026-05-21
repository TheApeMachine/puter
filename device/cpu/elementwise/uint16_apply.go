package elementwise

import "unsafe"

func runAddF16(dst, left, right unsafe.Pointer, count int) {
	addF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runSubF16(dst, left, right unsafe.Pointer, count int) {
	subF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMulF16(dst, left, right unsafe.Pointer, count int) {
	mulF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runDivF16(dst, left, right unsafe.Pointer, count int) {
	divF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMaxF16(dst, left, right unsafe.Pointer, count int) {
	maxF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMinF16(dst, left, right unsafe.Pointer, count int) {
	minF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runAbsF16(dst, src unsafe.Pointer, count int) {
	absF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runNegF16(dst, src unsafe.Pointer, count int) {
	negF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runSqrtF16(dst, src unsafe.Pointer, count int) {
	sqrtF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runReluF16(dst, src unsafe.Pointer, count int) {
	reluF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runAxpyF16(y, x unsafe.Pointer, count int, alpha float32) {
	axpyF16Kernel((*uint16)(y), (*uint16)(x), alpha, count)
}

func runAddBF16(dst, left, right unsafe.Pointer, count int) {
	addBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runSubBF16(dst, left, right unsafe.Pointer, count int) {
	subBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMulBF16(dst, left, right unsafe.Pointer, count int) {
	mulBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runDivBF16(dst, left, right unsafe.Pointer, count int) {
	divBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMaxBF16(dst, left, right unsafe.Pointer, count int) {
	maxBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runMinBF16(dst, left, right unsafe.Pointer, count int) {
	minBF16Kernel((*uint16)(dst), (*uint16)(left), (*uint16)(right), count)
}

func runAbsBF16(dst, src unsafe.Pointer, count int) {
	absBF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runNegBF16(dst, src unsafe.Pointer, count int) {
	negBF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runSqrtBF16(dst, src unsafe.Pointer, count int) {
	sqrtBF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runReluBF16(dst, src unsafe.Pointer, count int) {
	reluBF16Kernel((*uint16)(dst), (*uint16)(src), count)
}

func runAxpyBF16(y, x unsafe.Pointer, count int, alpha float32) {
	axpyBF16Kernel((*uint16)(y), (*uint16)(x), alpha, count)
}
