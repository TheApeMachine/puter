package reduction

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var sumFP16Kernel = func() func(values *uint16, count int) uint16 {
	return pickFP16SumKernel(sumFP16Funcs)
}()

var prodFP16Kernel = func() func(values *uint16, count int) float32 {
	return pickFP16ProdKernel(prodFP16Funcs)
}()

var minFP16Kernel = func() func(values *uint16, count int) float32 {
	return pickFP16MinKernel(minFP16Funcs)
}()

var maxFP16Kernel = func() func(values *uint16, count int) float32 {
	return pickFP16MaxKernel(maxFP16Funcs)
}()

var l1NormFP16Kernel = func() func(values *uint16, count int) float32 {
	return pickFP16L1NormKernel(l1NormFP16Funcs)
}()

func dispatchSumFP16(values unsafe.Pointer, count int) dtype.F16 {
	if count == 0 {
		return 0
	}

	return dtype.F16(sumFP16Kernel(
		(*uint16)(values),
		count,
	))
}

func SumFP16Generic(values *uint16, count int) uint16 {
	view := unsafe.Slice(values, count)
	var sum float32

	for index := 0; index < count; index++ {
		value := dtype.F16(view[index])
		sum += value.Float32()
	}

	return uint16(dtype.Fromfloat32(sum))
}

func dispatchProdFP16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return prodFP16Kernel(
		(*uint16)(values),
		count,
	)
}

func ProdFP16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	var product float32 = 1

	for index := 0; index < count; index++ {
		value := dtype.F16(view[index])
		product *= value.Float32()
	}

	return product
}

func dispatchMinFP16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return minFP16Kernel(
		(*uint16)(values),
		count,
	)
}

func MinFP16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	minimum := dtype.F16(view[0]).Float32()

	for index := 1; index < count; index++ {
		value := dtype.F16(view[index]).Float32()

		if value < minimum {
			minimum = value
		}
	}

	return minimum
}

func dispatchMaxFP16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return maxFP16Kernel(
		(*uint16)(values),
		count,
	)
}

func MaxFP16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	maximum := dtype.F16(view[0]).Float32()

	for index := 1; index < count; index++ {
		value := dtype.F16(view[index]).Float32()

		if value > maximum {
			maximum = value
		}
	}

	return maximum
}

func dispatchL1NormFP16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return l1NormFP16Kernel(
		(*uint16)(values),
		count,
	)
}

func L1NormFP16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	var norm float32

	for index := 0; index < count; index++ {
		value := dtype.F16(view[index]).Float32()
		norm += float32(math.Abs(float64(value)))
	}

	return norm
}
