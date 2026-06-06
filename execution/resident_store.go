package execution

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/types"
)

/*
ResidentStore maps safetensors parameter names to backend-resident tensors.
It satisfies WeightStore and the optional transpose/slice lookup interfaces
execution uses for projection.linear and packed QKV weights.
*/
type ResidentStore struct {
	memory     tensor.Backend
	tokens     map[string]types.Token
	tensors    map[string]tensor.Tensor
	transposed map[string]tensor.Tensor
	sliced     map[sliceCacheKey]tensor.Tensor
	sliceMu    sync.Mutex
}

type sliceCacheKey struct {
	name      string
	axis      string
	start     int64
	end       int64
	transpose bool
}

/*
NewResidentStore constructs an empty store backed by the given memory backend.
*/
func NewResidentStore(memory tensor.Backend) *ResidentStore {
	return &ResidentStore{
		memory:     memory,
		tokens:     make(map[string]types.Token),
		tensors:    make(map[string]tensor.Tensor),
		transposed: make(map[string]tensor.Tensor),
		sliced:     make(map[sliceCacheKey]tensor.Tensor),
	}
}

/*
RegisterTensor records one resident parameter tensor under its checkpoint name.
*/
func (store *ResidentStore) RegisterTensor(name string, token types.Token, resident tensor.Tensor) {
	if store == nil {
		return
	}

	store.tokens[name] = token
	store.tensors[name] = resident
}

/*
Lookup returns the resident tensor for one checkpoint parameter name.
*/
func (store *ResidentStore) Lookup(name string) (tensor.Tensor, error) {
	if store == nil {
		return nil, ErrWeightNotFound
	}

	resident, ok := store.tensors[name]

	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrWeightNotFound, name)
	}

	return resident, nil
}

/*
LookupTransposed returns the row-major transpose of a two-dimensional weight.
*/
func (store *ResidentStore) LookupTransposed(name string) (tensor.Tensor, error) {
	if store == nil {
		return nil, ErrWeightNotFound
	}

	if cached, ok := store.transposed[name]; ok {
		return cached, nil
	}

	base, err := store.Lookup(name)

	if err != nil {
		return nil, err
	}

	transposed, err := transposeMatrix(store.memory, base)

	if err != nil {
		return nil, err
	}

	store.transposed[name] = transposed

	return transposed, nil
}

/*
LookupSlice materializes a range from a packed checkpoint tensor.
*/
func (store *ResidentStore) LookupSlice(name, axis string, start, end int64) (tensor.Tensor, error) {
	return store.lookupSlice(name, axis, start, end, false)
}

/*
LookupTransposedSlice materializes a sliced range and returns its transpose.
*/
func (store *ResidentStore) LookupTransposedSlice(name, axis string, start, end int64) (tensor.Tensor, error) {
	return store.lookupSlice(name, axis, start, end, true)
}

func (store *ResidentStore) lookupSlice(
	name, axis string,
	start, end int64,
	transposed bool,
) (tensor.Tensor, error) {
	cacheKey := sliceCacheKey{
		name:      name,
		axis:      axis,
		start:     start,
		end:       end,
		transpose: transposed,
	}

	store.sliceMu.Lock()
	cached, ok := store.sliced[cacheKey]
	store.sliceMu.Unlock()

	if ok {
		return cached, nil
	}

	base, err := store.Lookup(name)

	if err != nil {
		return nil, err
	}

	sliced, err := sliceWeightTensor(store.memory, base, axis, start, end)

	if err != nil {
		return nil, err
	}

	if transposed {
		sliced, err = transposeMatrix(store.memory, sliced)

		if err != nil {
			return nil, err
		}
	}

	store.sliceMu.Lock()
	store.sliced[cacheKey] = sliced
	store.sliceMu.Unlock()

	return sliced, nil
}

/*
Close releases every registered resident tensor.
*/
func (store *ResidentStore) Close() error {
	if store == nil {
		return nil
	}

	for _, resident := range store.tensors {
		_ = resident.Close()
	}

	for _, resident := range store.transposed {
		_ = resident.Close()
	}

	for _, resident := range store.sliced {
		_ = resident.Close()
	}

	store.tensors = nil
	store.transposed = nil
	store.sliced = nil
	store.tokens = nil

	return nil
}

var (
	_ WeightStore            = (*ResidentStore)(nil)
	_ TransposedLookup       = (*ResidentStore)(nil)
	_ SliceLookup            = (*ResidentStore)(nil)
	_ TransposedSliceLookup  = (*ResidentStore)(nil)
)

func transposeMatrix(memory tensor.Backend, input tensor.Tensor) (tensor.Tensor, error) {
	dimensions := input.Shape().Dims()

	if len(dimensions) != 2 {
		return nil, fmt.Errorf("execution: transpose requires rank-2 weight, got %v", dimensions)
	}

	rows := dimensions[0]
	cols := dimensions[1]
	outputShape, err := tensor.NewShape([]int{cols, rows})

	if err != nil {
		return nil, err
	}

	byteCount, err := outputShape.Bytes(input.DType())

	if err != nil {
		return nil, err
	}

	rawBytes, err := readTensorBytes(memory, input)

	if err != nil {
		return nil, err
	}

	elementSize, err := input.DType().Size()

	if err != nil {
		return nil, err
	}

	outputBytes := make([]byte, byteCount)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			sourceOffset := (row*cols + col) * elementSize
			destinationOffset := (col*rows + row) * elementSize
			copy(
				outputBytes[destinationOffset:destinationOffset+elementSize],
				rawBytes[sourceOffset:sourceOffset+elementSize],
			)
		}
	}

	return memory.Upload(outputShape, input.DType(), outputBytes)
}

func sliceWeightTensor(
	memory tensor.Backend,
	input tensor.Tensor,
	axis string,
	start, end int64,
) (tensor.Tensor, error) {
	dimensions := input.Shape().Dims()

	if len(dimensions) != 2 {
		return nil, fmt.Errorf("execution: weight slice requires rank-2 tensor, got %v", dimensions)
	}

	if end <= start {
		return nil, fmt.Errorf("execution: invalid slice range [%d,%d)", start, end)
	}

	elementSize, err := input.DType().Size()

	if err != nil {
		return nil, err
	}

	if elementSize == 0 {
		return nil, fmt.Errorf("execution: unsupported dtype for weight slice: %s", input.DType())
	}

	rawBytes, err := readTensorBytes(memory, input)

	if err != nil {
		return nil, err
	}

	switch axis {
	case "output":
		cols := dimensions[1]
		rowBytes := cols * elementSize
		sliceRows := int(end - start)
		outputShape, shapeErr := tensor.NewShape([]int{sliceRows, cols})

		if shapeErr != nil {
			return nil, shapeErr
		}

		outputByteCount, byteErr := outputShape.Bytes(input.DType())

		if byteErr != nil {
			return nil, byteErr
		}

		outputBytes := make([]byte, outputByteCount)
		sourceOffset := int(start) * rowBytes

		copy(outputBytes, rawBytes[sourceOffset:sourceOffset+len(outputBytes)])

		return memory.Upload(outputShape, input.DType(), outputBytes)
	case "input":
		rows := dimensions[0]
		cols := dimensions[1]
		sliceCols := int(end - start)
		outputShape, shapeErr := tensor.NewShape([]int{rows, sliceCols})

		if shapeErr != nil {
			return nil, shapeErr
		}

		outputByteCount, byteErr := outputShape.Bytes(input.DType())

		if byteErr != nil {
			return nil, byteErr
		}

		outputBytes := make([]byte, outputByteCount)

		for row := 0; row < rows; row++ {
			sourceStart := (row*cols + int(start)) * elementSize
			destStart := row * sliceCols * elementSize
			copy(
				outputBytes[destStart:destStart+sliceCols*elementSize],
				rawBytes[sourceStart:sourceStart+sliceCols*elementSize],
			)
		}

		return memory.Upload(outputShape, input.DType(), outputBytes)
	default:
		return nil, fmt.Errorf("execution: unsupported weight slice axis %q", axis)
	}
}

func uploadTokenTensor(memory tensor.Backend, token types.Token, rawBytes []byte) (tensor.Tensor, error) {
	dimensions := make([]int, len(token.Shape))

	for index, dimension := range token.Shape {
		dimensions[index] = int(dimension)
	}

	shape, err := tensor.NewShape(dimensions)

	if err != nil {
		return nil, err
	}

	precision := token.Precision

	if precision == dtype.Invalid {
		return nil, fmt.Errorf("execution: tensor %q has invalid dtype", token.Name)
	}

	return memory.Upload(shape, precision, rawBytes)
}

func readTensorBytes(memory tensor.Backend, input tensor.Tensor) ([]byte, error) {
	_, rawBytes, err := input.RawBytes()

	if err == nil {
		return rawBytes, nil
	}

	_, downloaded, downloadErr := memory.Download(input)

	if downloadErr != nil {
		return nil, downloadErr
	}

	return downloaded, nil
}
