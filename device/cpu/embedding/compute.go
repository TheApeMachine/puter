package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func loadInt32(pointer unsafe.Pointer, index int) int32 {
	return *(*int32)(unsafe.Add(pointer, uintptr(index)*4))
}

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

func dispatchLookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	if indexCount == 0 || hidden == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		runLookupF32Native(table, indices, output, vocab, hidden, indexCount)
	case dtype.Float16, dtype.BFloat16:
		runLookupReduced(table, indices, output, vocab, hidden, indexCount, format)
	default:
		panic("embedding: unsupported dtype")
	}
}

func dispatchBag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	if bagCount == 0 || hidden == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		runBagF32Native(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)
	case dtype.Float16, dtype.BFloat16:
		runBagReduced(table, indices, offsets, output, vocab, hidden, bagCount, indexCount, format)
	default:
		panic("embedding: unsupported dtype")
	}
}

func runLookupReduced(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	for resultIndex := 0; resultIndex < indexCount; resultIndex++ {
		tokenID := int(loadInt32(indices, resultIndex))

		if tokenID < 0 || tokenID >= vocab {
			panic("embedding: index out of range")
		}

		for dimIndex := 0; dimIndex < hidden; dimIndex++ {
			tableIndex := tokenID*hidden + dimIndex
			outputIndex := resultIndex*hidden + dimIndex

			switch format {
			case dtype.Float16:
				bits := *(*uint16)(unsafe.Add(table, uintptr(tableIndex)*2))
				*(*uint16)(unsafe.Add(output, uintptr(outputIndex)*2)) = bits
			case dtype.BFloat16:
				bits := *(*uint16)(unsafe.Add(table, uintptr(tableIndex)*2))
				*(*uint16)(unsafe.Add(output, uintptr(outputIndex)*2)) = bits
			}
		}
	}
}

func runBagReduced(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	switch format {
	case dtype.Float16:
		runBagF16(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)
	case dtype.BFloat16:
		runBagBF16(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)
	default:
		panic("embedding: unsupported dtype")
	}
}

func runBagBF16(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
) {
	tableView := unsafe.Slice((*dtype.BF16)(table), vocab*hidden)
	outputView := unsafe.Slice((*dtype.BF16)(output), bagCount*hidden)
	scratch := make([]dtype.BF16, hidden)

	for bagIndex := 0; bagIndex < bagCount; bagIndex++ {
		startIdx := int(loadInt32(offsets, bagIndex))
		endIdx := indexCount

		if bagIndex+1 < bagCount {
			endIdx = int(loadInt32(offsets, bagIndex+1))
		}

		for dimIndex := range hidden {
			scratch[dimIndex] = 0
		}

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(loadInt32(indices, elementIndex))

			if tokenID < 0 || tokenID >= vocab {
				panic("embedding: index out of range")
			}

			row := tableView[tokenID*hidden : (tokenID+1)*hidden]
			addBF16InPlace(scratch, row)
		}

		outOffset := bagIndex * hidden
		copy(outputView[outOffset:outOffset+hidden], scratch)
	}
}

func runBagF16(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
) {
	tableView := unsafe.Slice((*dtype.F16)(table), vocab*hidden)
	outputView := unsafe.Slice((*dtype.F16)(output), bagCount*hidden)
	scratch := make([]dtype.F16, hidden)

	for bagIndex := 0; bagIndex < bagCount; bagIndex++ {
		startIdx := int(loadInt32(offsets, bagIndex))
		endIdx := indexCount

		if bagIndex+1 < bagCount {
			endIdx = int(loadInt32(offsets, bagIndex+1))
		}

		for dimIndex := range hidden {
			scratch[dimIndex] = 0
		}

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(loadInt32(indices, elementIndex))

			if tokenID < 0 || tokenID >= vocab {
				panic("embedding: index out of range")
			}

			row := tableView[tokenID*hidden : (tokenID+1)*hidden]
			addF16InPlace(scratch, row)
		}

		outOffset := bagIndex * hidden
		copy(outputView[outOffset:outOffset+hidden], scratch)
	}
}

func addBF16InPlace(accumulator, row []dtype.BF16) {
	for index := range accumulator {
		sum := (&accumulator[index]).Float32() + (&row[index]).Float32()
		accumulator[index] = dtype.NewBfloat16FromFloat32(sum)
	}
}

func addF16InPlace(accumulator, row []dtype.F16) {
	for index := range accumulator {
		sum := accumulator[index].Float32() + row[index].Float32()
		accumulator[index] = dtype.Fromfloat32(sum)
	}
}
