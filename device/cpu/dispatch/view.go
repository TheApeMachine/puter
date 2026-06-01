package dispatch

import (
	"slices"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

const viewMagic uint32 = 0x50555452

/*
View carries tensor metadata for CPU device.Backend calls that receive
only unsafe.Pointer values. Backends that pass raw data pointers must
still supply explicit element counts on methods that carry a count
parameter; View is required when a method infers sizes from tensor shape
(concat, checkpoint encode, gather row counts, and similar).
*/
type View struct {
	Magic        uint32
	Data         unsafe.Pointer
	ElementCount int
	Dims         []int
}

/*
WrapPointer returns an unsafe.Pointer to a heap-allocated View describing
raw storage. Callers must use the pointer only for the duration of the
synchronous device dispatch that follows.
*/
func WrapPointer(data unsafe.Pointer, elementCount int, dims []int) unsafe.Pointer {
	if data == nil {
		return nil
	}

	view := &View{
		Magic:        viewMagic,
		Data:         data,
		ElementCount: elementCount,
		Dims:         slices.Clone(dims),
	}

	return unsafe.Pointer(view)
}

/*
ResolvePointer unwraps a View or treats pointer as raw storage with unknown
length.
*/
func ResolvePointer(pointer unsafe.Pointer) (data unsafe.Pointer, elementCount int, dims []int, wrapped bool) {
	if pointer == nil {
		return nil, 0, nil, false
	}

	header := (*uint32)(pointer)

	if *header != viewMagic {
		return pointer, 0, nil, false
	}

	view := (*View)(pointer)

	if view.Data == nil {
		return pointer, 0, nil, false
	}

	return view.Data, view.ElementCount, view.Dims, true
}

/*
RequireElementCount returns the element count for pointer, using View when
present and otherwise requiring an explicit fallback count.
*/
func RequireElementCount(pointer unsafe.Pointer, fallbackCount int) int {
	_, elementCount, _, wrapped := ResolvePointer(pointer)

	if wrapped {
		return elementCount
	}

	if fallbackCount > 0 {
		return fallbackCount
	}

	panic("cpu dispatch: element count required")
}

/*
Int32Scalar reads one int32 from a device-resident scalar slot.
*/
func Int32Scalar(pointer unsafe.Pointer) int32 {
	if pointer == nil {
		panic("cpu dispatch: nil scalar pointer")
	}

	return *(*int32)(pointer)
}

/*
Float32Slice aliases count float32 lanes from pointer.
*/
func Float32Slice(pointer unsafe.Pointer, count int) []float32 {
	data, elementCount, _, wrapped := ResolvePointer(pointer)

	if wrapped && count <= 0 {
		count = elementCount
	}

	if count == 0 {
		return nil
	}

	return unsafe.Slice((*float32)(data), count)
}

/*
Uint8Slice aliases count bytes from pointer.
*/
func Uint8Slice(pointer unsafe.Pointer, count int) []byte {
	data, elementCount, _, wrapped := ResolvePointer(pointer)

	if wrapped && count <= 0 {
		count = elementCount
	}

	if count == 0 {
		return nil
	}

	return unsafe.Slice((*byte)(data), count)
}

/*
Int32Slice aliases count int32 lanes from pointer.
*/
func Int32Slice(pointer unsafe.Pointer, count int) []int32 {
	data, elementCount, _, wrapped := ResolvePointer(pointer)

	if wrapped && count <= 0 {
		count = elementCount
	}

	if count == 0 {
		return nil
	}

	return unsafe.Slice((*int32)(data), count)
}

/*
Uint16Slice aliases count uint16 lanes from pointer.
*/
func Uint16Slice(pointer unsafe.Pointer, count int) []uint16 {
	data, elementCount, _, wrapped := ResolvePointer(pointer)

	if wrapped && count <= 0 {
		count = elementCount
	}

	if count == 0 {
		return nil
	}

	return unsafe.Slice((*uint16)(data), count)
}

/*
ElementByteSize returns the byte width for supported shape/math dtypes.
*/
func ElementByteSize(format dtype.DType) int {
	switch format {
	case dtype.Float32, dtype.Int32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 0
	}
}

/*
RequireFloat32 panics when format is not float32.
*/
func RequireFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("cpu dispatch: float32 dtype required")
	}
}
