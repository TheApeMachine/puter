package runner

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
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

	if storageDType == dtype.Float32 {
		return cache.memory.Upload(shape, storageDType, raw)
	}

	if storageDType == dtype.Float16 || storageDType == dtype.BFloat16 {
		return cache.memory.Upload(shape, storageDType, raw)
	}

	if storageDType == dtype.Float64 {
		float32Values, convertErr := convert.BytesToFloat32(dtype.Float64, raw)

		if convertErr != nil {
			return nil, convertErr
		}

		return cache.memory.Upload(shape, dtype.Float32, convert.Float32ToBytes(float32Values))
	}

	return nil, fmt.Errorf("runner: unsupported weight dtype %s", storageDType)
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
