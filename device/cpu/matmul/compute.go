package matmul

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

type reducedLoadFunc func(pointer unsafe.Pointer, index int) float32

type reducedStoreFunc func(pointer unsafe.Pointer, index int, value float32)

func clearFloat32Matrix(out []float32, rows, cols int) {
	for index := range out[:rows*cols] {
		out[index] = 0
	}
}

func clearFloat64Matrix(out []float64, rows, cols int) {
	for index := range out[:rows*cols] {
		out[index] = 0
	}
}

func dispatchMatmul(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
	f32 func(out, left, right unsafe.Pointer, rows, inner, cols int),
) {
	switch format {
	case dtype.Float32:
		f32(out, left, right, rows, inner, cols)
	case dtype.Float64:
		runMatmulF64(out, left, right, rows, inner, cols)
	case dtype.Float16:
		MatmulFloat16Native(out, left, right, rows, inner, cols)
	case dtype.BFloat16:
		MatmulBFloat16Native(out, left, right, rows, inner, cols)
	}
}

func runMatmulReduced(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	load reducedLoadFunc,
	store reducedStoreFunc,
) {
	runMatmulReducedCols(out, left, right, rows, inner, cols, load, store, 0)
}

func runMatmulReducedCols(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	load reducedLoadFunc,
	store reducedStoreFunc,
	colStart int,
) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := colStart; colIndex < cols; colIndex++ {
			var sum float32

			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				leftValue := load(left, rowIndex*inner+innerIndex)
				rightValue := load(right, innerIndex*cols+colIndex)
				sum += leftValue * rightValue
			}

			store(out, rowIndex*cols+colIndex, sum)
		}
	}
}
