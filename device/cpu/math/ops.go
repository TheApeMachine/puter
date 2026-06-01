package math

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func (math Math) InvSqrtDimScale(
	out, input unsafe.Pointer,
	dim int32,
	format dtype.DType,
) {
	dispatch.RequireFloat32(format)

	outputData, count, _, wrapped := dispatch.ResolvePointer(out)

	if !wrapped {
		panic("math: InvSqrtDimScale requires dispatch.View on output")
	}

	inputData, inputCount, _, inputWrapped := dispatch.ResolvePointer(input)

	if !inputWrapped || inputCount != count {
		panic("math: InvSqrtDimScale input/output length mismatch")
	}

	InvSqrtDimScaleF32(
		dispatch.Float32Slice(outputData, count),
		dispatch.Float32Slice(inputData, count),
		dim,
	)
}

func (math Math) LogSumExp(
	input, output unsafe.Pointer,
	cols int,
	format dtype.DType,
) {
	dispatch.RequireFloat32(format)

	if cols <= 0 {
		panic("math: LogSumExp cols must be positive")
	}

	inputData, inputCount, _, inputWrapped := dispatch.ResolvePointer(input)

	if !inputWrapped || inputCount%cols != 0 {
		panic("math: LogSumExp input length must be divisible by cols")
	}

	outputData, outputCount, _, outputWrapped := dispatch.ResolvePointer(output)

	if !outputWrapped {
		panic("math: LogSumExp requires dispatch.View on output")
	}

	rows := inputCount / cols

	if outputCount != rows {
		panic("math: LogSumExp output row count mismatch")
	}

	LogSumExpF32(
		dispatch.Float32Slice(inputData, inputCount),
		cols,
		dispatch.Float32Slice(outputData, rows),
	)
}

func (math Math) Outer(
	left, right, output unsafe.Pointer,
	leftCount, rightCount int,
	format dtype.DType,
) {
	dispatch.RequireFloat32(format)

	if leftCount <= 0 || rightCount <= 0 {
		return
	}

	leftData, _, _, _ := dispatch.ResolvePointer(left)
	rightData, _, _, _ := dispatch.ResolvePointer(right)
	outputData, _, _, _ := dispatch.ResolvePointer(output)

	OuterF32(
		dispatch.Float32Slice(leftData, leftCount),
		dispatch.Float32Slice(rightData, rightCount),
		dispatch.Float32Slice(outputData, leftCount*rightCount),
	)
}
