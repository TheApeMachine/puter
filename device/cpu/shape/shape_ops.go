package shape

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Shape-manipulation kernels: gather, scatter, concat, split, expand,
transpose, masked_fill, where. These are the data-movement primitives
that live above the contiguous-storage contract. Strided rearrangements
materialize via these kernels.
*/

/*
Gather selects rows from a [N, D] source by indices [M] producing a
[M, D] output.
*/
func runGatherFloat32Int32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	sourceBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	sourceDims := args[0].Shape().Dims()

	if len(sourceDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	innerDim := sourceDims[1]
	innerBytes := innerDim * elementSize

	if len(outBytes) != len(indices)*innerBytes {
		return tensor.ErrShapeMismatch
	}

	for resultIndex, sourceRow := range indices {
		if int(sourceRow) < 0 || int(sourceRow) >= sourceDims[0] {
			return tensor.ErrShapeMismatch
		}

		copyContiguousElements(
			outBytes[resultIndex*innerBytes:(resultIndex+1)*innerBytes],
			sourceBytes[int(sourceRow)*innerBytes:(int(sourceRow)+1)*innerBytes],
			innerDim,
			elementSize,
		)
	}

	return nil
}

/*
Scatter writes rows from updates [M, D] to target [N, D] at indices
[M]. The args order is (target, indices, updates, output).
*/
func runScatterFloat32Int32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	targetBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	updatesBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[3])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	targetDims := args[0].Shape().Dims()

	if len(targetDims) != 2 || len(outBytes) != len(targetBytes) {
		return tensor.ErrShapeMismatch
	}

	innerDim := targetDims[1]
	innerBytes := innerDim * elementSize

	if len(updatesBytes) != len(indices)*innerBytes {
		return tensor.ErrShapeMismatch
	}

	copyContiguousElements(outBytes, targetBytes, len(outBytes)/elementSize, elementSize)

	for updateIndex, targetRow := range indices {
		if int(targetRow) < 0 || int(targetRow) >= targetDims[0] {
			return tensor.ErrShapeMismatch
		}

		copyContiguousElements(
			outBytes[int(targetRow)*innerBytes:(int(targetRow)+1)*innerBytes],
			updatesBytes[updateIndex*innerBytes:(updateIndex+1)*innerBytes],
			innerDim,
			elementSize,
		)
	}

	return nil
}

/*
Where selects entries from positive/negative based on a boolean mask:
out[i] = mask[i] ? positive[i] : negative[i].
*/
func runWhereFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	mask, err := args[0].BoolNative()

	if err != nil {
		return err
	}

	positiveBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	negativeBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[3])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[1])

	if err != nil {
		return err
	}

	elementCount := len(positiveBytes) / elementSize

	if len(negativeBytes) != len(positiveBytes) ||
		len(outBytes) != len(positiveBytes) ||
		mask.Len() != elementCount {
		return tensor.ErrShapeMismatch
	}

	maskBytes := bitVectorMaskBytes(mask)
	whereElements(outBytes, positiveBytes, negativeBytes, maskBytes, elementCount, elementSize)

	return nil
}

/*
MaskedFill replaces input entries where mask is true with the value
read from the scalar tensor (length-1 float32). Output preserves
input dtype/shape.
*/
func runMaskedFillFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	inputBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	mask, err := args[1].BoolNative()

	if err != nil {
		return err
	}

	fillBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[3])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	elementCount := len(inputBytes) / elementSize

	if len(outBytes) != len(inputBytes) ||
		mask.Len() != elementCount ||
		len(fillBytes) < elementSize {
		return tensor.ErrShapeMismatch
	}

	maskBytes := bitVectorMaskBytes(mask)
	maskedFillElements(outBytes, inputBytes, fillBytes, maskBytes, elementCount, elementSize)

	return nil
}

/*
Transpose2D swaps the two axes of a 2-D contiguous tensor.
*/
func runTranspose2DFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inputBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	inDims := args[0].Shape().Dims()
	outDims := args[1].Shape().Dims()

	if len(inDims) != 2 || len(outDims) != 2 ||
		inDims[0] != outDims[1] || inDims[1] != outDims[0] {
		return tensor.ErrShapeMismatch
	}

	rows := inDims[0]
	cols := inDims[1]

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			copyElementAt(outBytes, inputBytes, col*rows+row, row*cols+col, elementSize)
		}
	}

	return nil
}

/*
Concat concatenates two same-rank tensors along axis 0. Phase 8
expansion adds the general N-axis form; the host reference here
covers the most common case (concat-along-batch / concat-along-seq).
*/
func runConcatFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	leftBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	rightBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	if len(outBytes) != len(leftBytes)+len(rightBytes) {
		return tensor.ErrShapeMismatch
	}

	leftCount := len(leftBytes) / elementSize
	rightCount := len(rightBytes) / elementSize

	copyContiguousElements(outBytes[:len(leftBytes)], leftBytes, leftCount, elementSize)
	copyContiguousElements(outBytes[len(leftBytes):], rightBytes, rightCount, elementSize)

	return nil
}
