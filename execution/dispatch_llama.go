package execution

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
This file holds the dispatcher op handlers Llama-style models need on
top of the elementwise/matmul/norm/embedding set already in
dispatch_table.go:

  - shape.view_as_heads   ([B*T, hidden]      → [B*T, num_heads, head_dim])
  - shape.merge_heads     ([B*T, num_heads, head_dim] → [B*T, hidden])
  - shape.last_token      ([T, hidden]        → [1, hidden])
  - activation.swiglu     two-tensor SwiGLU (silu(gate) * up)
  - positional.rope       per-head RoPE rotation
  - attention.gqa         grouped-query SDPA (KV head count <= Q head count)

The shape ops are zero-kernel — they re-view the same byte buffer with a
new tensor.Shape via tensor.NewAliasedHostTensor. The kernel ops dispatch
through new methods on executionDevice (SwiGLUTensors, RoPE,
MultiHeadAttention) that the active device.Backend already implements
on the embedded family interfaces.

Registering these here keeps the new handlers reviewable in isolation
without bulking dispatch_table.go further. The init() at the bottom
plugs them into the shared opTable.
*/

/*
handleViewAsHeads turns [batch, hidden] into [batch, num_heads, head_dim]
by re-viewing the input's bytes with a new tensor.Shape. The element
count is unchanged; head_dim is derived as last_dim / num_heads.
*/
func handleViewAsHeads(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("shape.view_as_heads expects one input, got %d", len(node.Inputs))
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	numHeads := intAttr(node, "num_heads", 0)

	if numHeads <= 0 {
		return fmt.Errorf("shape.view_as_heads requires positive num_heads config, got %d", numHeads)
	}

	dims := input.Shape().Dims()

	if len(dims) < 1 {
		return fmt.Errorf("shape.view_as_heads input has rank 0")
	}

	lastDim := dims[len(dims)-1]

	if lastDim%numHeads != 0 {
		return fmt.Errorf(
			"shape.view_as_heads: last dim %d is not divisible by num_heads %d",
			lastDim, numHeads,
		)
	}

	headDim := lastDim / numHeads

	newDims := append(append([]int(nil), dims[:len(dims)-1]...), numHeads, headDim)

	output, err := reviewTensor(input, newDims)

	if err != nil {
		return fmt.Errorf("shape.view_as_heads: %w", err)
	}

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleMergeHeads is the inverse of view_as_heads: [batch, num_heads,
head_dim] → [batch, num_heads * head_dim]. No config needed — the head
collapse is derived from the input's trailing two dims.
*/
func handleMergeHeads(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("shape.merge_heads expects one input, got %d", len(node.Inputs))
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	dims := input.Shape().Dims()

	if len(dims) < 2 {
		return fmt.Errorf("shape.merge_heads input must have rank >= 2, got %d", len(dims))
	}

	numHeads := dims[len(dims)-2]
	headDim := dims[len(dims)-1]
	merged := numHeads * headDim

	newDims := append(append([]int(nil), dims[:len(dims)-2]...), merged)

	output, err := reviewTensor(input, newDims)

	if err != nil {
		return fmt.Errorf("shape.merge_heads: %w", err)
	}

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleLastToken slices off the final position along the sequence axis:
[seq, hidden] → [1, hidden]. Used at the end of a Llama forward pass
right before lm_head to project only the most recent token. Implemented
as a copy of the trailing per-row stride rather than a view because
later ops (the lm_head Matmul) own their own output and won't mutate the
slice, but a zero-copy view of the tail of a larger buffer is brittle
when downstream allocations get reused.
*/
func handleLastToken(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("shape.last_token expects one input, got %d", len(node.Inputs))
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	dims := input.Shape().Dims()

	if len(dims) < 2 {
		return fmt.Errorf("shape.last_token input must have rank >= 2, got %d", len(dims))
	}

	rows := dims[0]
	hidden := dims[len(dims)-1]

	if rows < 1 {
		return fmt.Errorf("shape.last_token: empty input (rows=%d)", rows)
	}

	values, err := input.Float32Native()

	if err != nil {
		return fmt.Errorf("shape.last_token: read input: %w", err)
	}

	lastRowBase := (rows - 1) * hidden

	if lastRowBase+hidden > len(values) {
		return fmt.Errorf(
			"shape.last_token: tail [%d, %d) exceeds value count %d",
			lastRowBase, lastRowBase+hidden, len(values),
		)
	}

	outputBytes := make([]byte, hidden*4)

	for index := 0; index < hidden; index++ {
		binary.LittleEndian.PutUint32(outputBytes[index*4:], math.Float32bits(values[lastRowBase+index]))
	}

	outputShape, err := tensor.NewShape([]int{1, hidden})

	if err != nil {
		return fmt.Errorf("shape.last_token: build output shape: %w", err)
	}

	output, err := dispatcher.memory.Upload(outputShape, dtype.Float32, outputBytes)

	if err != nil {
		return fmt.Errorf("shape.last_token: upload: %w", err)
	}

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleSwiGLU runs activation.swiglu over two same-shape inputs (gate, up)
and writes silu(gate) * up to the output. Llama's MLP uses this between
gate_proj/up_proj and down_proj.
*/
func handleSwiGLU(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 2 {
		return fmt.Errorf("activation.swiglu expects two inputs, got %d", len(node.Inputs))
	}

	gate, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	up, err := dispatcher.values.tensor(node.Inputs[1])

	if err != nil {
		return err
	}

	count := gate.Len()

	if count != up.Len() {
		return fmt.Errorf(
			"activation.swiglu: gate has %d elements, up has %d",
			count, up.Len(),
		)
	}

	// Bypass allocateOutput — the planner's workspace tensor for this
	// node carries the typer's bestEffortPassthrough shape (gate's
	// shape, which happens to be correct here) but the same is NOT
	// true for view_as_heads / merge_heads / rope / gqa where the
	// typer passthrough lies. Until typer specs land for these ops
	// (task #10), allocating fresh keeps downstream consumers honest
	// — they read shape from this tensor, not from the workspace.
	output, err := dispatcher.memory.Upload(gate.Shape(), dtype.Float32, make([]byte, count*4))

	if err != nil {
		return err
	}

	gatePtr, _, err := pointerOf(gate)

	if err != nil {
		return err
	}

	upPtr, _, err := pointerOf(up)

	if err != nil {
		return err
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.SwiGLUTensors(outputPtr, gatePtr, upPtr, count, dtype.Float32)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleRoPE applies rotary position embeddings to per-head Q or K tensors
shaped [seqLen, numHeads, headDim]. Config attributes:

  - base / rope_theta : base frequency (Llama-3 uses 500000)
  - head_dim          : last-dim size
  - start_position    : optional, for KV-cache continuation

Llama-3-specific frequency scaling (rope_type=llama3, rope_factor etc.)
is ignored for the first cut — the underlying device.RoPE.RoPE method
only takes BaseFreq + StartPosition. Adding Llama-3 scaling means
either pre-scaling the frequency table in this handler or extending
device.RoPEConfig with the four scaling fields.
*/
func handleRoPE(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("positional.rope expects one input, got %d", len(node.Inputs))
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	dims := input.Shape().Dims()

	if len(dims) < 3 {
		return fmt.Errorf(
			"positional.rope expects rank>=3 input [seq, heads, head_dim], got %v",
			dims,
		)
	}

	seqLen := dims[0]
	numHeads := dims[len(dims)-2]
	headDim := dims[len(dims)-1]

	baseFreq := floatAttr(node, "base", 0)

	if baseFreq == 0 {
		baseFreq = floatAttr(node, "rope_theta", 10000)
	}

	startPosition := intAttr(node, "start_position", 0)

	config := device.RoPEConfig{
		BaseFreq:      baseFreq,
		StartPosition: startPosition,
	}

	// See the comment in handleSwiGLU about bypassing allocateOutput
	// until typer specs land for the Llama-specific ops. Here we
	// allocate fresh with the input's [seq, num_heads, head_dim]
	// shape so shape.merge_heads downstream can derive num_heads /
	// head_dim from this tensor's trailing dims.
	output, err := dispatcher.memory.Upload(input.Shape(), dtype.Float32, make([]byte, seqLen*numHeads*headDim*4))

	if err != nil {
		return err
	}

	inputPtr, _, err := pointerOf(input)

	if err != nil {
		return err
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.RoPE(config, inputPtr, outputPtr, seqLen, numHeads, headDim, dtype.Float32)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleGQA runs grouped-query attention via the unified
MultiHeadAttention device call. Llama-3.2-1B has 32 query heads and
8 KV heads (KVHeadCount=8). Inputs in topological order:

  0. query [seqQ, num_heads, head_dim]
  1. key   [seqK, num_kv_heads, head_dim]
  2. value [seqK, num_kv_heads, head_dim]

Config attributes: num_heads, num_kv_heads, head_dim, causal.
Output shape matches query.
*/
func handleGQA(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 3 {
		return fmt.Errorf("attention.gqa expects three inputs (q, k, v), got %d", len(node.Inputs))
	}

	query, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	key, err := dispatcher.values.tensor(node.Inputs[1])

	if err != nil {
		return err
	}

	value, err := dispatcher.values.tensor(node.Inputs[2])

	if err != nil {
		return err
	}

	qDims := query.Shape().Dims()
	kDims := key.Shape().Dims()

	if len(qDims) < 3 || len(kDims) < 3 {
		return fmt.Errorf(
			"attention.gqa: expected rank>=3 q/k tensors, got q=%v k=%v",
			qDims, kDims,
		)
	}

	seqQ := qDims[0]
	seqK := kDims[0]

	numHeads := intAttr(node, "num_heads", qDims[len(qDims)-2])
	numKVHeads := intAttr(node, "num_kv_heads", kDims[len(kDims)-2])
	headDim := intAttr(node, "head_dim", qDims[len(qDims)-1])
	causal := boolAttr(node, "causal", true)

	config := device.MultiHeadAttentionConfig{
		NumHeads:    numHeads,
		HeadDim:     headDim,
		Causal:      causal,
		KVHeadCount: numKVHeads,
	}

	// Build the output's shape from config rather than from
	// query.Shape() — query may carry the typer's passthrough rank-2
	// shape rather than the actual [seq, num_heads, head_dim]. Same
	// rationale as handleSwiGLU's allocateOutput bypass.
	outputShape, err := tensor.NewShape([]int{seqQ, numHeads, headDim})

	if err != nil {
		return fmt.Errorf("attention.gqa: build output shape: %w", err)
	}

	output, err := dispatcher.memory.Upload(outputShape, dtype.Float32, make([]byte, seqQ*numHeads*headDim*4))

	if err != nil {
		return err
	}

	queryPtr, _, err := pointerOf(query)

	if err != nil {
		return err
	}

	keyPtr, _, err := pointerOf(key)

	if err != nil {
		return err
	}

	valuePtr, _, err := pointerOf(value)

	if err != nil {
		return err
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.MultiHeadAttention(config, queryPtr, keyPtr, valuePtr, outputPtr, seqQ, seqK, dtype.Float32)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
reviewTensor wraps the input tensor's bytes in a new HostTensor with the
requested shape. Element count must match — caller's responsibility.
Used by shape.view_as_heads and shape.merge_heads; both are pure shape
ops with no kernel work, so re-viewing avoids a copy through Upload.
*/
func reviewTensor(input tensor.Tensor, newDims []int) (tensor.Tensor, error) {
	host, ok := input.(*tensor.HostTensor)

	if !ok {
		return nil, fmt.Errorf(
			"shape op: input is %T, only host-resident tensors are re-viewable today",
			input,
		)
	}

	storageDType, rawBytes, err := host.RawBytes()

	if err != nil {
		return nil, fmt.Errorf("shape op: read input bytes: %w", err)
	}

	newShape, err := tensor.NewShape(newDims)

	if err != nil {
		return nil, fmt.Errorf("shape op: build target shape %v: %w", newDims, err)
	}

	return tensor.NewAliasedHostTensor(newShape, storageDType, rawBytes), nil
}

/*
floatAttr reads a numeric config attribute as float64, accepting any of
int / int64 / float32 / float64 / json.Number-equivalent. Returns the
fallback when the attribute is missing or unrecognized.
*/
func floatAttr(node *ast.GraphNode, key string, fallback float64) float64 {
	if node == nil || node.Attributes == nil {
		return fallback
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return fallback
	}

	switch typed := raw.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return fallback
	}
}

/*
boolAttr reads a boolean config attribute. YAML decoders return either
bool or string for boolean-shaped values, so accept both.
*/
func boolAttr(node *ast.GraphNode, key string, fallback bool) bool {
	if node == nil || node.Attributes == nil {
		return fallback
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return fallback
	}

	switch typed := raw.(type) {
	case bool:
		return typed
	case string:
		switch typed {
		case "true", "True", "TRUE", "yes", "1":
			return true
		case "false", "False", "FALSE", "no", "0":
			return false
		}
	}

	return fallback
}

/*
init plugs the Llama-specific handlers into the shared opTable defined
in dispatch_table.go. Keeping them in this file means a reviewer can see
the whole Llama op surface in one place without re-reading the
universal table.
*/
func init() {
	opTable["shape.view_as_heads"] = handleViewAsHeads
	opTable["shape.merge_heads"] = handleMergeHeads
	opTable["shape.last_token"] = handleLastToken
	opTable["activation.swiglu"] = handleSwiGLU
	opTable["positional.rope"] = handleRoPE
	opTable["attention.gqa"] = handleGQA
}
