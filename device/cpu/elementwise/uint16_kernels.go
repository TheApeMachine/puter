package elementwise

var (
	addF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(addF16Funcs)
	}()

	subF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(subF16Funcs)
	}()

	mulF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(mulF16Funcs)
	}()

	divF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(divF16Funcs)
	}()

	maxF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(maxF16Funcs)
	}()

	minF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(minF16Funcs)
	}()

	absF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(absF16Funcs)
	}()

	negF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(negF16Funcs)
	}()

	sqrtF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(sqrtF16Funcs)
	}()

	reluF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(reluF16Funcs)
	}()

	axpyF16Kernel = func() func(y, x *uint16, alpha float32, count int) {
		return pickUInt16AxpyKernel(axpyF16Funcs)
	}()

	addBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(addBF16Funcs)
	}()

	subBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(subBF16Funcs)
	}()

	mulBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(mulBF16Funcs)
	}()

	divBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(divBF16Funcs)
	}()

	maxBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(maxBF16Funcs)
	}()

	minBF16Kernel = func() func(dst, left, right *uint16, count int) {
		return pickUInt16BinaryKernel(minBF16Funcs)
	}()

	absBF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(absBF16Funcs)
	}()

	negBF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(negBF16Funcs)
	}()

	sqrtBF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(sqrtBF16Funcs)
	}()

	reluBF16Kernel = func() func(dst, src *uint16, count int) {
		return pickUInt16UnaryKernel(reluBF16Funcs)
	}()

	axpyBF16Kernel = func() func(y, x *uint16, alpha float32, count int) {
		return pickUInt16AxpyKernel(axpyBF16Funcs)
	}()
)
