package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/shape"
)

type pageIntrinsicDevice interface {
	PageWrite(
		storage, values, pageIDs, offsets, output unsafe.Pointer,
		pageCount, pageSize, inner, valueRows, storageOffset int,
		format dtype.DType,
	)
	PageGather(
		storage, pageTable, output unsafe.Pointer,
		pageCount, pageSize, inner, outRows, storageOffset int,
		format dtype.DType,
	)
}

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

	stateStorage := storage

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

	if stateStorage.Location() != tensor.Host {
		return runPageWriteDeviceIntrinsic(
			resolver,
			stateStorage,
			values,
			pageIDs,
			offsets,
			pageSize,
			layerIndex,
		)
	}

	layerStorage, err := layerStorageView(stateStorage, layerIndex)

	if err != nil {
		return err
	}

	pageSizeTensor, err := uploadScalarInt32(resolver.dispatcher.memory, pageSize)

	if err != nil {
		return err
	}

	if err := shape.RunPageWrite(
		layerStorage,
		values,
		pageIDs,
		offsets,
		pageSizeTensor,
		layerStorage,
	); err != nil {
		return err
	}

	resolver.storeOutput(stateStorage)

	return nil
}

func runPageWriteDeviceIntrinsic(
	resolver *bindResolver,
	storage tensor.Tensor,
	values tensor.Tensor,
	pageIDs tensor.Tensor,
	offsets tensor.Tensor,
	pageSize int,
	layerIndex int,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(pageIntrinsicDevice)

	if !ok {
		return fmt.Errorf(
			"state.page_write: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			storage.Location(),
		)
	}

	config, err := pageDeviceConfig(storage, values, pageSize, layerIndex)

	if err != nil {
		return err
	}

	if pageIDs.Len() != offsets.Len() || pageIDs.Len() > config.valueRows {
		return tensor.ErrShapeMismatch
	}

	storagePointer, _, err := pointerOf(storage)

	if err != nil {
		return err
	}

	valuesPointer, _, err := pointerOf(values)

	if err != nil {
		return err
	}

	pageIDsPointer, _, err := pointerOf(pageIDs)

	if err != nil {
		return err
	}

	offsetsPointer, _, err := pointerOf(offsets)

	if err != nil {
		return err
	}

	deviceBackend.PageWrite(
		storagePointer,
		valuesPointer,
		pageIDsPointer,
		offsetsPointer,
		storagePointer,
		config.pageCount,
		pageSize,
		config.inner,
		pageIDs.Len(),
		config.storageOffset,
		storage.DType(),
	)

	resolver.storeOutput(storage)

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

	if storage.Location() != tensor.Host {
		return runPageGatherDeviceIntrinsic(
			resolver,
			storage,
			pageTable,
			pageSize,
			layerIndex,
		)
	}

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

	resolver.storeOutput(output)

	return nil
}

func runPageGatherDeviceIntrinsic(
	resolver *bindResolver,
	storage tensor.Tensor,
	pageTable tensor.Tensor,
	pageSize int,
	layerIndex int,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(pageIntrinsicDevice)

	if !ok {
		return fmt.Errorf(
			"state.page_gather: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			storage.Location(),
		)
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return err
	}

	config, err := pageDeviceConfig(storage, output, pageSize, layerIndex)

	if err != nil {
		return err
	}

	outRows := config.valueRows
	liveRows := liveKVLength(resolver.dispatcher.launchBindings)

	if liveRows > 0 && liveRows < outRows {
		outRows = liveRows
	}

	maxRows := pageTable.Len() * pageSize

	if maxRows < outRows {
		outRows = maxRows
	}

	storagePointer, _, err := pointerOf(storage)

	if err != nil {
		return err
	}

	pageTablePointer, _, err := pointerOf(pageTable)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	deviceBackend.PageGather(
		storagePointer,
		pageTablePointer,
		outputPointer,
		config.pageCount,
		pageSize,
		config.inner,
		outRows,
		config.storageOffset,
		storage.DType(),
	)

	resolver.storeOutput(output)

	return nil
}

type pageDeviceKernelConfig struct {
	pageCount     int
	inner         int
	valueRows     int
	storageOffset int
}

func pageDeviceConfig(
	storage tensor.Tensor,
	values tensor.Tensor,
	pageSize int,
	layerIndex int,
) (pageDeviceKernelConfig, error) {
	storageDims := storage.Shape().Dims()
	valueDims := values.Shape().Dims()
	storageOffset := 0

	if len(storageDims) == 5 {
		if layerIndex < 0 || layerIndex >= storageDims[0] {
			return pageDeviceKernelConfig{}, tensor.ErrShapeMismatch
		}

		layerElements := productInts(storageDims[1:])
		storageOffset = layerIndex * layerElements
		storageDims = storageDims[1:]
	}

	if len(storageDims) < 2 || len(valueDims) != len(storageDims)-1 {
		return pageDeviceKernelConfig{}, tensor.ErrShapeMismatch
	}

	if storageDims[1] != pageSize || storage.DType() != values.DType() {
		return pageDeviceKernelConfig{}, tensor.ErrShapeMismatch
	}

	inner, err := trailingElementCount(valueDims[1:], storageDims[2:])

	if err != nil {
		return pageDeviceKernelConfig{}, err
	}

	return pageDeviceKernelConfig{
		pageCount:     storageDims[0],
		inner:         inner,
		valueRows:     valueDims[0],
		storageOffset: storageOffset,
	}, nil
}

func trailingElementCount(valueDims []int, storageDims []int) (int, error) {
	if len(valueDims) != len(storageDims) {
		return 0, tensor.ErrShapeMismatch
	}

	count := 1

	for index, valueDim := range valueDims {
		if valueDim != storageDims[index] {
			return 0, tensor.ErrShapeMismatch
		}

		count *= valueDim
	}

	return count, nil
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
