//go:build xla

package xla

import (
	"context"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
DeviceTensor is XLA-resident storage returned from Upload and workspace allocation.
*/
type DeviceTensor struct {
	backend       *Backend
	shape         tensor.Shape
	elementFormat dtype.DType
	buffer        C.XLABufferRef
	byteCount     int
	state         atomic.Uint32
	closed        atomic.Bool
	mu            sync.Mutex
	cond          *sync.Cond
}

func newDeviceTensor(
	backend *Backend,
	shape tensor.Shape,
	elementFormat dtype.DType,
	buffer C.XLABufferRef,
	byteCount int,
	pending bool,
) *DeviceTensor {
	deviceTensor := &DeviceTensor{
		backend:       backend,
		shape:         shape,
		elementFormat: elementFormat,
		buffer:        buffer,
		byteCount:     byteCount,
	}

	deviceTensor.cond = sync.NewCond(&deviceTensor.mu)

	if pending {
		deviceTensor.state.Store(uint32(tensor.StatePending))
		return deviceTensor
	}

	deviceTensor.state.Store(uint32(tensor.StateReady))
	return deviceTensor
}

func (deviceTensor *DeviceTensor) Shape() tensor.Shape {
	return deviceTensor.shape
}

func (deviceTensor *DeviceTensor) DType() dtype.DType {
	return deviceTensor.elementFormat
}

func (deviceTensor *DeviceTensor) Layout() tensor.Layout {
	return tensor.LayoutDense
}

func (deviceTensor *DeviceTensor) Location() tensor.Location {
	return tensor.XLA
}

func (deviceTensor *DeviceTensor) Len() int {
	return deviceTensor.shape.Len()
}

func (deviceTensor *DeviceTensor) Bytes() int {
	return deviceTensor.byteCount
}

func (deviceTensor *DeviceTensor) State() tensor.State {
	return tensor.State(deviceTensor.state.Load())
}

func (deviceTensor *DeviceTensor) WaitReady() error {
	if deviceTensor.closed.Load() {
		return tensor.ErrTensorClosed
	}

	if deviceTensor.State() != tensor.StatePending {
		return nil
	}

	deviceTensor.state.Store(uint32(tensor.StateReady))
	return nil
}

func (deviceTensor *DeviceTensor) Sync(ctx context.Context) error {
	if err := deviceTensor.WaitReady(); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	return nil
}

func (deviceTensor *DeviceTensor) Close() error {
	if !deviceTensor.closed.CompareAndSwap(false, true) {
		return nil
	}

	if waitErr := deviceTensor.WaitReady(); waitErr != nil {
		return waitErr
	}

	if deviceTensor.buffer != nil && deviceTensor.backend != nil && deviceTensor.backend.bridge != nil {
		deviceTensor.backend.bridge.releaseBuffer(deviceTensor.buffer)
		deviceTensor.buffer = nil
	}

	deviceTensor.state.Store(uint32(tensor.StateClosed))
	return nil
}

func (deviceTensor *DeviceTensor) RawBytes() (dtype.DType, []byte, error) {
	if deviceTensor.closed.Load() {
		return dtype.Invalid, nil, tensor.ErrTensorClosed
	}

	if waitErr := deviceTensor.WaitReady(); waitErr != nil {
		return dtype.Invalid, nil, waitErr
	}

	if deviceTensor.backend == nil || deviceTensor.backend.bridge == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	_, bytesOut, err := deviceTensor.backend.bridge.download(deviceTensor)
	return deviceTensor.elementFormat, bytesOut, err
}

func (deviceTensor *DeviceTensor) Slice(start, length int) (tensor.Tensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (deviceTensor *DeviceTensor) Reshape(dims []int) (tensor.Tensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (deviceTensor *DeviceTensor) Float64Native() ([]float64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Float32Native() ([]float32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Float16Native() ([]dtype.F16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) BFloat16Native() ([]dtype.BF16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Int64Native() ([]int64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Int32Native() ([]int32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Int16Native() ([]int16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Int8Native() ([]int8, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Uint64Native() ([]uint64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Uint32Native() ([]uint32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Uint16Native() ([]uint16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Uint8Native() ([]uint8, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) BoolNative() (tensor.BitVector, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) Int4Native() (tensor.Int4Vector, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (deviceTensor *DeviceTensor) bufferRef() C.XLABufferRef {
	return deviceTensor.buffer
}

func (deviceTensor *DeviceTensor) format() dtype.DType {
	return deviceTensor.elementFormat
}

func (deviceTensor *DeviceTensor) storageBytes() int64 {
	return int64(deviceTensor.byteCount)
}

func (deviceTensor *DeviceTensor) residentToken() uintptr {
	return uintptr(unsafe.Pointer(deviceTensor))
}
