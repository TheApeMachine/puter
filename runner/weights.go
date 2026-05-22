package runner

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/weights"
)

/*
weightCache loads checkpoint tensors once per weights file.
*/
type weightCache struct {
	memory tensor.Backend
	mu     sync.Mutex
	tables map[string]map[string]tensor.Tensor
}

func newWeightCache(memory tensor.Backend) *weightCache {
	return &weightCache{
		memory: memory,
		tables: make(map[string]map[string]tensor.Tensor),
	}
}

func (cache *weightCache) Tensor(path string, tensorName string) (tensor.Tensor, error) {
	if path == "" || tensorName == "" {
		return nil, fmt.Errorf("runner: weight path and tensor name are required")
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	fileTable, ok := cache.tables[path]

	if !ok {
		fileTable = make(map[string]tensor.Tensor)
		cache.tables[path] = fileTable
	}

	if cached, ok := fileTable[tensorName]; ok {
		return cached, nil
	}

	resident, err := cache.loadTensor(path, tensorName)

	if err != nil {
		return nil, err
	}

	fileTable[tensorName] = resident

	return resident, nil
}

func (cache *weightCache) TensorForNode(
	path string,
	tensorName string,
	node *ir.Node,
) (tensor.Tensor, error) {
	weightSlice, ok := weightSliceFromNode(node)
	if !ok {
		return cache.Tensor(path, tensorName)
	}

	if path == "" || tensorName == "" {
		return nil, fmt.Errorf("runner: weight path and tensor name are required")
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	fileTable, ok := cache.tables[path]

	if !ok {
		fileTable = make(map[string]tensor.Tensor)
		cache.tables[path] = fileTable
	}

	cacheKey := fmt.Sprintf(
		"%s|%s:%d:%d",
		tensorName,
		weightSlice.Axis,
		weightSlice.Start,
		weightSlice.End,
	)

	if cached, ok := fileTable[cacheKey]; ok {
		return cached, nil
	}

	resident, err := cache.loadTensorSlice(path, tensorName, node, weightSlice)

	if err != nil {
		return nil, err
	}

	fileTable[cacheKey] = resident

	return resident, nil
}

func (cache *weightCache) Close() {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	for _, fileTable := range cache.tables {
		for _, value := range fileTable {
			_ = value.Close()
		}
	}

	cache.tables = make(map[string]map[string]tensor.Tensor)
}

func (cache *weightCache) loadTensor(path string, tensorName string) (tensor.Tensor, error) {
	raw, meta, err := readWeightBytes(path, tensorName)

	if err != nil {
		return nil, err
	}

	storageDType, err := dtype.Parse(meta.DType)

	if err != nil {
		return nil, err
	}

	shape, err := shapeFromMeta(meta.Shape)

	if err != nil {
		return nil, err
	}

	return uploadWeightBytes(cache.memory, shape, storageDType, raw)
}

func (cache *weightCache) loadTensorSlice(
	path string,
	tensorName string,
	node *ir.Node,
	weightSlice astWeightSlice,
) (tensor.Tensor, error) {
	raw, meta, err := readWeightBytes(path, tensorName)

	if err != nil {
		return nil, err
	}

	storageDType, err := dtype.Parse(meta.DType)

	if err != nil {
		return nil, err
	}

	raw, meta, err = sliceWeightBytes(raw, meta, storageDType, node, weightSlice)

	if err != nil {
		return nil, err
	}

	shape, err := shapeFromMeta(meta.Shape)

	if err != nil {
		return nil, err
	}

	return uploadWeightBytes(cache.memory, shape, storageDType, raw)
}

func uploadWeightBytes(
	memory tensor.Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
	raw []byte,
) (tensor.Tensor, error) {
	if storageDType == dtype.Float32 {
		return memory.Upload(shape, storageDType, raw)
	}

	if storageDType == dtype.Float16 || storageDType == dtype.BFloat16 {
		return memory.Upload(shape, storageDType, raw)
	}

	if storageDType == dtype.Float64 {
		float32Values, convertErr := convert.BytesToFloat32(dtype.Float64, raw)

		if convertErr != nil {
			return nil, convertErr
		}

		return memory.Upload(shape, dtype.Float32, convert.Float32ToBytes(float32Values))
	}

	return nil, fmt.Errorf("runner: unsupported weight dtype %s", storageDType)
}

type astWeightSlice struct {
	Axis  string
	Start int64
	End   int64
}

func weightSliceFromNode(node *ir.Node) (astWeightSlice, bool) {
	if node == nil {
		return astWeightSlice{}, false
	}

	metadata := node.Metadata()
	axis, ok := metadata["weight_slice_axis"].(string)

	if !ok || axis == "" {
		return astWeightSlice{}, false
	}

	return astWeightSlice{
		Axis:  axis,
		Start: int64FromMetadata(metadata["weight_slice_start"]),
		End:   int64FromMetadata(metadata["weight_slice_end"]),
	}, true
}

func int64FromMetadata(value any) int64 {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int64:
		return typed
	case float64:
		return int64(typed)
	default:
		return 0
	}
}

func sliceWeightBytes(
	raw []byte,
	meta weights.TensorMeta,
	storageDType dtype.DType,
	node *ir.Node,
	weightSlice astWeightSlice,
) ([]byte, weights.TensorMeta, error) {
	if len(meta.Shape) != 2 {
		return nil, meta, fmt.Errorf("runner: sliced weight %q must be 2D", node.ID())
	}

	elementBytes, err := storageDType.Size()
	if err != nil {
		return nil, meta, err
	}

	axis, err := weightSliceAxis(weightSlice.Axis)
	if err != nil {
		return nil, meta, err
	}

	start := int(weightSlice.Start)
	end, err := weightSliceEnd(node, weightSlice, axis, meta.Shape)

	if err != nil {
		return nil, meta, err
	}

	if start < 0 || end < start || end > int(meta.Shape[axis]) {
		return nil, meta, fmt.Errorf("runner: weight slice [%d:%d] out of range for axis %d", start, end, axis)
	}

	rows := int(meta.Shape[0])
	columns := int(meta.Shape[1])
	sliceLen := end - start

	if axis == 0 {
		offset := start * columns * elementBytes
		limit := end * columns * elementBytes
		meta.Shape[0] = int64(sliceLen)

		return append([]byte(nil), raw[offset:limit]...), meta, nil
	}

	out := make([]byte, rows*sliceLen*elementBytes)

	for row := range rows {
		sourceOffset := (row*columns + start) * elementBytes
		sourceLimit := sourceOffset + sliceLen*elementBytes
		targetOffset := row * sliceLen * elementBytes

		copy(out[targetOffset:], raw[sourceOffset:sourceLimit])
	}

	meta.Shape[1] = int64(sliceLen)

	return out, meta, nil
}

func weightSliceAxis(axis string) (int, error) {
	switch axis {
	case "output":
		return 0, nil
	case "input":
		return 1, nil
	default:
		return 0, fmt.Errorf("runner: unsupported weight slice axis %q", axis)
	}
}

func weightSliceEnd(
	node *ir.Node,
	weightSlice astWeightSlice,
	axis int,
	shape []int64,
) (int, error) {
	if weightSlice.End > 0 {
		return int(weightSlice.End), nil
	}

	attributeName := "out_features"
	if axis == 1 {
		attributeName = "in_features"
	}

	length, err := nodeIntAttribute(node, attributeName)

	if err != nil {
		return int(shape[axis]), nil
	}

	return int(weightSlice.Start) + length, nil
}

func shapeFromMeta(dimensions []int64) (tensor.Shape, error) {
	dims := make([]int, len(dimensions))

	for index, dimension := range dimensions {
		if dimension < 0 {
			return tensor.Shape{}, fmt.Errorf("invalid weight dimension %d", dimension)
		}

		dims[index] = int(dimension)
	}

	return tensor.NewShape(dims)
}

func zeroTensor(
	memory tensor.Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	byteCount, err := storageDType.BytesFor(shape.Len())

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, storageDType, make([]byte, byteCount))
}
