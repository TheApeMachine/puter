package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/shape"
)

func liveKVLength(bindings ir.SymbolMap) int {
	if bindings == nil {
		return 0
	}

	liveValue, ok := bindings["KV"]

	if !ok {
		return 0
	}

	return int(liveValue)
}

func isPageIntrinsicMethod(method string) bool {
	switch method {
	case "state.page_write", "state.page_gather":
		return true
	default:
		return false
	}
}

func runPageIntrinsic(resolver *bindResolver) error {
	switch resolver.bind.Method {
	case "state.page_write":
		return runPageWriteIntrinsic(resolver)
	case "state.page_gather":
		return runPageGatherIntrinsic(resolver)
	default:
		return fmt.Errorf("unknown page intrinsic %q", resolver.bind.Method)
	}
}

func runPageWriteIntrinsic(resolver *bindResolver) error {
	storage, err := resolver.resolveInputTensor("0")

	if err != nil {
		return err
	}

	values, err := resolver.resolveInputTensor("1")

	if err != nil {
		return err
	}

	pageIDs, err := resolver.resolveInputTensor("2")

	if err != nil {
		return err
	}

	offsets, err := resolver.resolveInputTensor("3")

	if err != nil {
		return err
	}

	pageSize := configInt(resolver.node, "page_size", 0)

	if pageSize <= 0 {
		return fmt.Errorf("state.page_write: page_size must be positive")
	}

	layerIndex := configInt(resolver.node, "layer_index", -1)

	storage, err = layerStorageView(storage, layerIndex)

	if err != nil {
		return err
	}

	pageSizeTensor, err := uploadScalarInt32(resolver.dispatcher.memory, pageSize)

	if err != nil {
		return err
	}

	if err := shape.RunPageWrite(storage, values, pageIDs, offsets, pageSizeTensor, storage); err != nil {
		return err
	}

	resolver.dispatcher.values.set(resolver.node.ID, storage)

	return nil
}

func runPageGatherIntrinsic(resolver *bindResolver) error {
	storage, err := resolver.resolveInputTensor("0")

	if err != nil {
		return err
	}

	pageTable, err := resolver.resolveInputTensor("1")

	if err != nil {
		return err
	}

	pageSize := configInt(resolver.node, "page_size", 0)

	if pageSize <= 0 {
		return fmt.Errorf("state.page_gather: page_size must be positive")
	}

	layerIndex := configInt(resolver.node, "layer_index", -1)

	storage, err = layerStorageView(storage, layerIndex)

	if err != nil {
		return err
	}

	pageSizeTensor, err := uploadScalarInt32(resolver.dispatcher.memory, pageSize)

	if err != nil {
		return err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return err
	}

	if err := shape.RunPageGatherWithLiveRows(storage, pageTable, pageSizeTensor, output, liveKVLength(resolver.dispatcher.launchBindings)); err != nil {
		return err
	}

	resolver.dispatcher.values.set(resolver.node.ID, output)

	return nil
}

func uploadScalarInt32(memory tensor.Backend, value int) (tensor.Tensor, error) {
	shapeDims, err := tensor.NewShape([]int{1})

	if err != nil {
		return nil, err
	}

	buffer := []byte{
		byte(value),
		byte(value >> 8),
		byte(value >> 16),
		byte(value >> 24),
	}

	return memory.Upload(shapeDims, dtype.Int32, buffer)
}
