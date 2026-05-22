package reduction

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var sumBF16Kernel = func() func(values *uint16, count int) uint16 {
	return pickBF16SumKernel(sumBF16Funcs)
}()

var prodBF16Kernel = func() func(values *uint16, count int) float32 {
	return pickBF16ProdKernel(prodBF16Funcs)
}()

var minBF16Kernel = func() func(values *uint16, count int) float32 {
	return pickBF16MinKernel(minBF16Funcs)
}()

var maxBF16Kernel = func() func(values *uint16, count int) float32 {
	return pickBF16MaxKernel(maxBF16Funcs)
}()

var l1NormBF16Kernel = func() func(values *uint16, count int) float32 {
	return pickBF16L1NormKernel(l1NormBF16Funcs)
}()

func dispatchSumBF16(values unsafe.Pointer, count int) dtype.BF16 {
	if count == 0 {
		return 0
	}

	return dtype.BF16(sumBF16Kernel(
		(*uint16)(values),
		count,
	))
}

func SumBF16Generic(values *uint16, count int) uint16 {
	view := unsafe.Slice(values, count)
	var sum float32

	for index := 0; index < count; index++ {
		value := dtype.BF16(view[index])
		sum += (&value).Float32()
	}

	return uint16(dtype.NewBfloat16FromFloat32(sum))
}

func dispatchProdBF16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return prodBF16Kernel(
		(*uint16)(values),
		count,
	)
}

func ProdBF16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	var product float32 = 1

	for index := 0; index < count; index++ {
		value := dtype.BF16(view[index])
		product *= (&value).Float32()
	}

	return product
}

func dispatchMinBF16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return minBF16Kernel(
		(*uint16)(values),
		count,
	)
}

func MinBF16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	first := dtype.BF16(view[0])
	minimum := (&first).Float32()

	for index := 1; index < count; index++ {
		value := dtype.BF16(view[index])
		converted := (&value).Float32()

		if converted < minimum {
			minimum = converted
		}
	}

	return minimum
}

func dispatchMaxBF16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return maxBF16Kernel(
		(*uint16)(values),
		count,
	)
}

func MaxBF16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	first := dtype.BF16(view[0])
	maximum := (&first).Float32()

	for index := 1; index < count; index++ {
		value := dtype.BF16(view[index])
		converted := (&value).Float32()

		if converted > maximum {
			maximum = converted
		}
	}

	return maximum
}

func dispatchL1NormBF16(values unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	return l1NormBF16Kernel(
		(*uint16)(values),
		count,
	)
}

func L1NormBF16Generic(values *uint16, count int) float32 {
	view := unsafe.Slice(values, count)
	var norm float32

	for index := 0; index < count; index++ {
		value := dtype.BF16(view[index])
		norm += float32(math.Abs(float64((&value).Float32())))
	}

	return norm
}
