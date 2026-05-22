package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

/*
Mixed-precision dispatch for multi_head_attention, grouped_query_attention,
and sliding_window_attention.
*/

func configForVariant(name string) MultiHeadAttentionConfig {
	config := DefaultMultiHeadAttentionConfig()

	switch name {
	case "grouped_query_attention":
		config.KVHeadCount = config.NumHeads / 4
	case "sliding_window_attention":
		config.Causal = true
		config.WindowSize = 128
	}

	return config
}

func multiHeadAttentionMixedDims(
	config MultiHeadAttentionConfig,
	query, key, value, out tensor.Tensor,
) (seqQ, seqK, kvHeads int, err error) {
	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	kvHeads = config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim

	if queryDims[1] != queryFeatures || keyDims[1] != kvFeatures ||
		valueDims[1] != kvFeatures || outDims[1] != queryFeatures {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	seqQ = queryDims[0]
	seqK = keyDims[0]

	if valueDims[0] != seqK || outDims[0] != seqQ {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return seqQ, seqK, kvHeads, nil
}

func runMultiHeadAttentionBFloat16(args []tensor.Tensor, config MultiHeadAttentionConfig) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	seqQ, seqK, kvHeads, err := multiHeadAttentionMixedDims(config, args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	queryNative, err := args[0].BFloat16Native()
	if err != nil {
		return err
	}

	keyNative, err := args[1].BFloat16Native()
	if err != nil {
		return err
	}

	valueNative, err := args[2].BFloat16Native()
	if err != nil {
		return err
	}

	outputNative, err := args[3].BFloat16Native()
	if err != nil {
		return err
	}

	multiHeadAttentionSlices(
		config,
		unsafe.Pointer(&queryNative[0]),
		unsafe.Pointer(&keyNative[0]),
		unsafe.Pointer(&valueNative[0]),
		unsafe.Pointer(&outputNative[0]),
		seqQ, seqK, kvHeads,
		args[0].DType(),
	)

	return nil
}

func runMultiHeadAttentionFloat16(args []tensor.Tensor, config MultiHeadAttentionConfig) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	seqQ, seqK, kvHeads, err := multiHeadAttentionMixedDims(config, args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	queryNative, err := args[0].Float16Native()
	if err != nil {
		return err
	}

	keyNative, err := args[1].Float16Native()
	if err != nil {
		return err
	}

	valueNative, err := args[2].Float16Native()
	if err != nil {
		return err
	}

	outputNative, err := args[3].Float16Native()
	if err != nil {
		return err
	}

	multiHeadAttentionSlices(
		config,
		unsafe.Pointer(&queryNative[0]),
		unsafe.Pointer(&keyNative[0]),
		unsafe.Pointer(&valueNative[0]),
		unsafe.Pointer(&outputNative[0]),
		seqQ, seqK, kvHeads,
		args[0].DType(),
	)

	return nil
}
