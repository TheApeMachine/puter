package shape

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Additional shape-manipulation kernels found in the canonical
transformer pipeline: last_token, merge_heads, split_heads, slice,
reshape (metadata-only), split.

  - last_token: extracts the final timestep from a [batch, seq, hidden]
    tensor.
  - merge_heads: collapses [batch, seq, heads, headDim] →
    [batch, seq, heads × headDim].
  - split_heads: inverse of merge_heads.
  - slice: copies a contiguous range out of the source.
  - split: splits a tensor into two halves along axis 0.
*/

func runLastToken(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inBytes, err := aliasedBytes(args[0])

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

	if len(inDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	seq := inDims[1]
	hidden := inDims[2]
	hiddenBytes := hidden * elementSize

	if len(outBytes) != batch*hiddenBytes {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		lastTokenOffset := ((batchIndex*seq + seq - 1) * hidden) * elementSize
		outOffset := batchIndex * hiddenBytes

		copyContiguousElements(
			outBytes[outOffset:outOffset+hiddenBytes],
			inBytes[lastTokenOffset:lastTokenOffset+hiddenBytes],
			hidden,
			elementSize,
		)
	}

	return nil
}

func runMergeHeads(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inBytes, err := aliasedBytes(args[0])

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

	if len(inDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	seq := inDims[1]
	heads := inDims[2]
	headDim := inDims[3]
	headDimBytes := headDim * elementSize

	if len(outBytes) != batch*seq*heads*headDimBytes {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for seqIndex := 0; seqIndex < seq; seqIndex++ {
			for headIndex := 0; headIndex < heads; headIndex++ {
				inOffset := (((batchIndex*seq+seqIndex)*heads + headIndex) * headDim) * elementSize
				outOffset := (((batchIndex*seq + seqIndex) * heads * headDim) + headIndex*headDim) * elementSize

				copyContiguousElements(
					outBytes[outOffset:outOffset+headDimBytes],
					inBytes[inOffset:inOffset+headDimBytes],
					headDim,
					elementSize,
				)
			}
		}
	}

	return nil
}

func runSplitHeads(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	outDims := args[1].Shape().Dims()

	if len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := outDims[0]
	seq := outDims[1]
	heads := outDims[2]
	headDim := outDims[3]
	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	if len(inBytes) != batch*seq*heads*headDim*elementSize {
		return tensor.ErrShapeMismatch
	}

	copyContiguousElements(outBytes, inBytes, len(inBytes)/elementSize, elementSize)

	return nil
}

func runSplit2(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	inBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	leftBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	rightBytes, err := aliasedBytes(args[2])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(args[0])

	if err != nil {
		return err
	}

	if len(inBytes) != len(leftBytes)+len(rightBytes) {
		return tensor.ErrShapeMismatch
	}

	leftCount := len(leftBytes) / elementSize
	rightCount := len(rightBytes) / elementSize

	copyContiguousElements(leftBytes, inBytes[:len(leftBytes)], leftCount, elementSize)
	copyContiguousElements(rightBytes, inBytes[len(leftBytes):], rightCount, elementSize)

	return nil
}
