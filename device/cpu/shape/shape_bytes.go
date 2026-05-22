package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func aliasedBytes(arg tensor.Tensor) ([]byte, error) {
	totalBytes := arg.Bytes()

	if totalBytes == 0 {
		return nil, nil
	}

	switch arg.DType() {
	case dtype.Float32:
		view, err := arg.Float32Native()

		if err != nil {
			return nil, err
		}

		return unsafe.Slice((*byte)(unsafe.Pointer(&view[0])), totalBytes), nil
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()

		if err != nil {
			return nil, err
		}

		return unsafe.Slice((*byte)(unsafe.Pointer(&view[0])), totalBytes), nil
	case dtype.Float16:
		view, err := arg.Float16Native()

		if err != nil {
			return nil, err
		}

		return unsafe.Slice((*byte)(unsafe.Pointer(&view[0])), totalBytes), nil
	default:
		return nil, tensor.ErrDTypeMismatch
	}
}

func elementByteSize(arg tensor.Tensor) (int, error) {
	return arg.DType().Size()
}

func copyContiguousElements(dst, src []byte, elementCount, elementSize int) {
	if elementCount == 0 {
		return
	}

	byteCount := elementCount * elementSize

	if elementSize == 4 {
		dstView := unsafe.Slice((*float32)(unsafe.Pointer(&dst[0])), elementCount)
		srcView := unsafe.Slice((*float32)(unsafe.Pointer(&src[0])), elementCount)
		CopyContiguousFloat32Native(dstView, srcView)

		return
	}

	copy(dst[:byteCount], src[:byteCount])
}

func whereElements(
	dst, positive, negative []byte,
	mask []byte,
	elementCount, elementSize int,
) {
	if elementSize == 4 {
		dstView := unsafe.Slice((*float32)(unsafe.Pointer(&dst[0])), elementCount)
		positiveView := unsafe.Slice((*float32)(unsafe.Pointer(&positive[0])), elementCount)
		negativeView := unsafe.Slice((*float32)(unsafe.Pointer(&negative[0])), elementCount)
		WhereFloat32Native(dstView, positiveView, negativeView, mask)

		return
	}

	for index := 0; index < elementCount; index++ {
		offset := index * elementSize

		if maskBitAt(mask, index) {
			copy(dst[offset:offset+elementSize], positive[offset:offset+elementSize])
			continue
		}

		copy(dst[offset:offset+elementSize], negative[offset:offset+elementSize])
	}
}

func maskedFillElements(
	dst, input, fill []byte,
	mask []byte,
	elementCount, elementSize int,
) {
	if elementSize == 4 {
		dstView := unsafe.Slice((*float32)(unsafe.Pointer(&dst[0])), elementCount)
		inputView := unsafe.Slice((*float32)(unsafe.Pointer(&input[0])), elementCount)
		fillView := unsafe.Slice((*float32)(unsafe.Pointer(&fill[0])), 1)
		MaskedFillFloat32Native(dstView, inputView, fillView[0], mask)

		return
	}

	for index := 0; index < elementCount; index++ {
		offset := index * elementSize

		copy(dst[offset:offset+elementSize], input[offset:offset+elementSize])

		if maskBitAt(mask, index) {
			copy(dst[offset:offset+elementSize], fill[:elementSize])
		}
	}
}

func copyElementAt(
	dst, src []byte,
	dstIndex, srcIndex, elementSize int,
) {
	dstOffset := dstIndex * elementSize
	srcOffset := srcIndex * elementSize

	copy(dst[dstOffset:dstOffset+elementSize], src[srcOffset:srcOffset+elementSize])
}
