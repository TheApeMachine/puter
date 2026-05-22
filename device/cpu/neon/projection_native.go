package neon

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/matmul"
)

func runLinear(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	storageDType := args[0].DType()

	if storageDType != args[1].DType() ||
		storageDType != args[2].DType() ||
		storageDType != args[3].DType() {
		return tensor.ErrDTypeMismatch
	}

	switch storageDType {
	case dtype.Float32:
		return runLinearFloat32(args...)
	case dtype.Float16, dtype.BFloat16:
		return runLinearReducedPrecision(args, storageDType)
	default:
		return tensor.ErrDTypeMismatch
	}
}

func runLinearFloat32(args ...tensor.Tensor) error {
	xView, _ := args[0].Float32Native()
	wView, _ := args[1].Float32Native()
	bView, _ := args[2].Float32Native()
	yView, _ := args[3].Float32Native()

	batch, inDim, outDim, err := linearDims(args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := bView[outIndex]

			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += xView[batchIndex*inDim+inIndex] *
					wView[outIndex*inDim+inIndex]
			}

			yView[batchIndex*outDim+outIndex] = sum
		}
	}

	return nil
}

func runLinearReducedPrecision(args []tensor.Tensor, format dtype.DType) error {
	xPointer, err := floatTensorPointer(args[0], format)

	if err != nil {
		return err
	}

	wPointer, err := floatTensorPointer(args[1], format)

	if err != nil {
		return err
	}

	bPointer, err := floatTensorPointer(args[2], format)

	if err != nil {
		return err
	}

	yPointer, err := floatTensorPointer(args[3], format)

	if err != nil {
		return err
	}

	batch, inDim, outDim, err := linearDims(args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		xRow := unsafe.Add(xPointer, uintptr(batchIndex*inDim)*elementStride(format))

		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := loadFloatElement(bPointer, outIndex, format)
			wRow := unsafe.Add(wPointer, uintptr(outIndex*inDim)*elementStride(format))

			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += loadFloatElement(xRow, inIndex, format) *
					loadFloatElement(wRow, inIndex, format)
			}

			yIndex := batchIndex*outDim + outIndex
			storeFloatElement(yPointer, yIndex, format, sum)
		}
	}

	return nil
}

func runMatMulAddReducedPrecision(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	format := args[0].DType()

	if format != args[1].DType() ||
		format != args[2].DType() ||
		format != args[3].DType() {
		return tensor.ErrDTypeMismatch
	}

	if format == dtype.Float32 {
		return runMatMulAdd(args...)
	}

	leftPointer, err := floatTensorPointer(args[0], format)

	if err != nil {
		return err
	}

	rightPointer, err := floatTensorPointer(args[1], format)

	if err != nil {
		return err
	}

	biasPointer, err := floatTensorPointer(args[2], format)

	if err != nil {
		return err
	}

	outPointer, err := floatTensorPointer(args[3], format)

	if err != nil {
		return err
	}

	aDims := args[0].Shape().Dims()
	bDims := args[1].Shape().Dims()

	rows := aDims[0]
	inner := aDims[1]
	cols := bDims[1]

	matmul.Matmul(outPointer, leftPointer, rightPointer, rows, inner, cols, format)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowBase := unsafe.Add(outPointer, uintptr(rowIndex*cols)*elementStride(format))

		for colIndex := 0; colIndex < cols; colIndex++ {
			value := loadFloatElement(rowBase, colIndex, format) +
				loadFloatElement(biasPointer, colIndex, format)
			storeFloatElement(rowBase, colIndex, format, value)
		}
	}

	return nil
}

func linearDims(
	input, weight, bias, output tensor.Tensor,
) (batch, inDim, outDim int, err error) {
	xDims := input.Shape().Dims()
	wDims := weight.Shape().Dims()
	bDims := bias.Shape().Dims()
	yDims := output.Shape().Dims()

	if len(xDims) != 2 || len(wDims) != 2 ||
		len(bDims) != 1 || len(yDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	batch = xDims[0]
	inDim = xDims[1]
	outDim = wDims[0]

	if wDims[1] != inDim || bDims[0] != outDim ||
		yDims[0] != batch || yDims[1] != outDim {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return batch, inDim, outDim, nil
}

func floatTensorPointer(value tensor.Tensor, format dtype.DType) (unsafe.Pointer, error) {
	switch format {
	case dtype.Float16:
		view, err := value.Float16Native()

		if err != nil {
			return nil, err
		}

		if len(view) == 0 {
			return unsafe.Pointer(nil), nil
		}

		return unsafe.Pointer(&view[0]), nil
	case dtype.BFloat16:
		view, err := value.BFloat16Native()

		if err != nil {
			return nil, err
		}

		if len(view) == 0 {
			return unsafe.Pointer(nil), nil
		}

		return unsafe.Pointer(&view[0]), nil
	default:
		return nil, tensor.ErrDTypeMismatch
	}
}

func elementStride(format dtype.DType) uintptr {
	switch format {
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 4
	}
}

func loadFloatElement(pointer unsafe.Pointer, index int, format dtype.DType) float32 {
	switch format {
	case dtype.Float16:
		bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
		return dtype.Frombits(bits).Float32()
	case dtype.BFloat16:
		bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
		bf16 := dtype.BF16(bits)
		return (&bf16).Float32()
	default:
		return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
	}
}

func storeFloatElement(pointer unsafe.Pointer, index int, format dtype.DType, value float32) {
	switch format {
	case dtype.Float16:
		bits := dtype.Fromfloat32(value).Bits()
		*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
	case dtype.BFloat16:
		encoded := dtype.NewBfloat16FromFloat32(value)
		*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
	default:
		*(*float32)(unsafe.Add(pointer, uintptr(index)*4)) = value
	}
}
