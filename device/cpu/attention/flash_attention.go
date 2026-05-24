package attention

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/activation"
)

/*
FlashAttention — block-tiled attention with online softmax. The host
reference here implements the algorithm faithfully but in scalar Go
without block-tiling optimization; the value of the reference is
that it matches the device implementations bit-for-bit so parity
tests in later sessions can validate the device kernels against it.

Args order: (query, key, value, output) with optional mask handling
via a separate FlashAttentionConfig parameter.

Tensor shapes (no batch dim — the batched variant lands in a follow-
up session):
  - query  [seqQ, depth]
  - key    [seqK, depth]
  - value  [seqK, valueDim]
  - output [seqQ, valueDim]

Per Phase 8.2, this is the same forward result as the basic attention
kernel; the difference is the block-tiled execution path that
reduces memory bandwidth on real hardware. The reference here is a
single-block execution to keep the math obvious.
*/

type FlashAttentionConfig = device.FlashAttentionConfig

func DefaultFlashAttentionConfig() FlashAttentionConfig {
	return FlashAttentionConfig{BlockSize: 64, Causal: false}
}

func runFlashAttentionBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	queryNative, err := query.BFloat16Native()
	if err != nil {
		return err
	}

	keyNative, err := key.BFloat16Native()
	if err != nil {
		return err
	}

	valueNative, err := value.BFloat16Native()
	if err != nil {
		return err
	}

	outputNative, err := out.BFloat16Native()
	if err != nil {
		return err
	}

	scaledDotProductAttention(
		DefaultFlashAttentionConfig(),
		unsafe.Pointer(&queryNative[0]),
		unsafe.Pointer(&keyNative[0]),
		unsafe.Pointer(&valueNative[0]),
		unsafe.Pointer(&outputNative[0]),
		seqQ, seqK, depth, valueDim,
		query.DType(),
	)

	return nil
}

func runFlashAttentionFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	queryNative, err := query.Float16Native()
	if err != nil {
		return err
	}

	keyNative, err := key.Float16Native()
	if err != nil {
		return err
	}

	valueNative, err := value.Float16Native()
	if err != nil {
		return err
	}

	outputNative, err := out.Float16Native()
	if err != nil {
		return err
	}

	scaledDotProductAttention(
		DefaultFlashAttentionConfig(),
		unsafe.Pointer(&queryNative[0]),
		unsafe.Pointer(&keyNative[0]),
		unsafe.Pointer(&valueNative[0]),
		unsafe.Pointer(&outputNative[0]),
		seqQ, seqK, depth, valueDim,
		query.DType(),
	)

	return nil
}

func runFlashAttentionFloat32Default(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return RunFlashAttentionFloat32(
		DefaultFlashAttentionConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
RunFlashAttentionFloat32 runs flash-attention with the supplied
configuration. Causal masking zeros the attention score for any
query→key pair where keyIndex > queryIndex (lower-triangular).
*/
func RunFlashAttentionFloat32(
	config FlashAttentionConfig,
	query, key, value, out tensor.Tensor,
) error {
	queryView, keyView, valueView, outView, seqQ, seqK, depth, valueDim, err :=
		attentionViews(query, key, value, out)

	if err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(
			queryView, keyView, valueView, outView,
			rowIndex, seqK, depth, valueDim, scale, config.Causal,
		)
	}

	return nil
}

func flashExpFloat32(value float32) float32 {
	if math.IsInf(float64(value), -1) || value < -88 {
		return 0
	}

	if math.IsInf(float64(value), 1) {
		return float32(math.Inf(1))
	}

	scratch := [1]float32{value}
	activation.New().Exp(
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		1,
		dtype.Float32,
	)

	return scratch[0]
}
