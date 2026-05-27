//go:build darwin && cgo

package metal

import (
	"context"
	_ "embed"
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

/*
#cgo CFLAGS: -I${SRCDIR}/internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "internal/bridge/core.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"

//go:generate go run ./internal/metallibgen

//go:embed kernels.metallib
var kernelsMetalLibrary []byte

type metalBridge struct {
	device  C.MetalDeviceRef
	backend *Backend
}

func openMetalBridge(backend *Backend) (*metalBridge, error) {
	if len(kernelsMetalLibrary) == 0 {
		return nil, fmt.Errorf("%w: empty Metal library", tensor.ErrNeedsPlatformSetup)
	}

	status := C.MetalStatus{}
	device := C.metal_open_default_device(
		(*C.uint8_t)(unsafe.Pointer(&kernelsMetalLibrary[0])),
		C.longlong(len(kernelsMetalLibrary)),
		&status,
	)
	runtime.KeepAlive(kernelsMetalLibrary)

	if device == nil {
		return nil, fmt.Errorf("%w: %s", tensor.ErrNeedsPlatformSetup, bridgeStatusMessage(status))
	}

	return &metalBridge{
		device:  device,
		backend: backend,
	}, nil
}

func (bridge *metalBridge) contextRef() uintptr {
	return uintptr(unsafe.Pointer(bridge.device))
}

func (bridge *metalBridge) waitIdle() {
	if bridge == nil || bridge.device == nil {
		return
	}

	C.metal_device_wait_idle(bridge.device)
}

func (bridge *metalBridge) beginBatch() error {
	if bridge == nil || bridge.device == nil {
		return tensor.ErrNeedsPlatformSetup
	}

	C.metal_layer_begin(bridge.device)
	return nil
}

func (bridge *metalBridge) endBatch() error {
	if bridge == nil || bridge.device == nil {
		return tensor.ErrNeedsPlatformSetup
	}

	status := C.MetalStatus{}
	code := C.metal_layer_end(bridge.device, &status)

	if code != 0 {
		return fmt.Errorf("metal batch: %s", bridgeStatusMessage(status))
	}

	return nil
}

/*
SyncDevice waits for in-flight Metal command buffers to finish.
*/
func (backend *Backend) SyncDevice() {
	if backend == nil || backend.bridge == nil {
		return
	}

	backend.bridge.waitIdle()
}

func (backend *Backend) BeginBatch() error {
	if backend == nil || backend.bridge == nil {
		return tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.beginBatch()
}

func (backend *Backend) EndBatch() error {
	if backend == nil || backend.bridge == nil {
		return tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.endBatch()
}

func (bridge *metalBridge) recommendedMaxWorkingSet() int64 {
	if bridge == nil || bridge.device == nil {
		return 0
	}

	return int64(C.metal_recommended_max_working_set(bridge.device))
}

func (bridge *metalBridge) close() error {
	if bridge.device != nil {
		C.metal_device_release(bridge.device)
		bridge.device = nil
	}

	return nil
}

func (bridge *metalBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	byteCount, err := shape.Bytes(sourceDType)

	if err != nil {
		return nil, err
	}

	if byteCount != len(bytesIn) {
		return nil, tensor.ErrShapeMismatch
	}

	var buffer C.MetalBufferRef

	if byteCount > 0 {
		buffer = C.metal_buffer_new_shared(bridge.device, C.longlong(byteCount))

		if buffer == nil {
			return nil, tensor.ErrAllocatorExhausted
		}

		contents := C.metal_buffer_contents(buffer)

		if contents == nil {
			C.metal_buffer_release(buffer)
			return nil, tensor.ErrNeedsPlatformSetup
		}

		C.memcpy(contents, unsafe.Pointer(&bytesIn[0]), C.size_t(byteCount))
	}

	deviceTensor := newDeviceTensor(bridge.backend, shape, sourceDType, buffer, byteCount)
	return deviceTensor, nil
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	deviceTensor, err := requireDeviceTensor(input)

	if err != nil {
		return dtype.Invalid, nil, err
	}

	if deviceTensor.byteCount == 0 {
		return deviceTensor.elementFormat, []byte{}, nil
	}

	bridge.waitIdle()

	contents := C.metal_buffer_contents(deviceTensor.buffer)

	if contents == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	bytesOut := make([]byte, deviceTensor.byteCount)
	C.memcpy(unsafe.Pointer(&bytesOut[0]), contents, C.size_t(deviceTensor.byteCount))

	return deviceTensor.elementFormat, bytesOut, nil
}

func bridgeStatusMessage(status C.MetalStatus) string {
	if status.code == 0 {
		return "metal bridge ok"
	}

	return C.GoString(&status.message[0])
}

/*
DeviceTensor is Metal-resident storage returned from Upload.
*/
type DeviceTensor struct {
	backend       *Backend
	shape         tensor.Shape
	elementFormat dtype.DType
	buffer        C.MetalBufferRef
	byteCount     int
	ownsBuffer    bool
	workspaceView bool
	state         atomic.Uint32
	closed        atomic.Bool
	gradFlag      atomic.Bool
}

func newDeviceTensor(
	backend *Backend,
	shape tensor.Shape,
	elementFormat dtype.DType,
	buffer C.MetalBufferRef,
	byteCount int,
) *DeviceTensor {
	deviceTensor := &DeviceTensor{
		backend:       backend,
		shape:         shape,
		elementFormat: elementFormat,
		buffer:        buffer,
		byteCount:     byteCount,
		ownsBuffer:    true,
	}

	deviceTensor.state.Store(uint32(tensor.StateReady))
	return deviceTensor
}

func requireDeviceTensor(input tensor.Tensor) (*DeviceTensor, error) {
	if input == nil {
		return nil, tensor.ErrTensorClosed
	}

	deviceTensor, ok := input.(*DeviceTensor)

	if !ok {
		return nil, fmt.Errorf("tensor: expected metal-resident tensor, got location %q", input.Location())
	}

	if deviceTensor.closed.Load() {
		return nil, tensor.ErrTensorClosed
	}

	return deviceTensor, nil
}

func resolveDeviceTensor(pointer unsafe.Pointer) *DeviceTensor {
	if pointer == nil {
		return nil
	}

	deviceTensor := (*DeviceTensor)(pointer)

	if deviceTensor.buffer == nil {
		return nil
	}

	return deviceTensor
}

func resolveBufferRef(pointer unsafe.Pointer) C.MetalBufferRef {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor != nil {
		return deviceTensor.buffer
	}

	return C.MetalBufferRef(pointer)
}

func (deviceTensor *DeviceTensor) Shape() tensor.Shape       { return deviceTensor.shape }
func (deviceTensor *DeviceTensor) DType() dtype.DType        { return deviceTensor.elementFormat }
func (deviceTensor *DeviceTensor) Layout() tensor.Layout     { return tensor.LayoutDense }
func (deviceTensor *DeviceTensor) Location() tensor.Location { return tensor.Metal }
func (deviceTensor *DeviceTensor) Len() int                  { return deviceTensor.shape.Len() }
func (deviceTensor *DeviceTensor) Bytes() int                { return deviceTensor.byteCount }
func (deviceTensor *DeviceTensor) State() tensor.State {
	return tensor.State(deviceTensor.state.Load())
}

func (deviceTensor *DeviceTensor) WaitReady() error {
	if deviceTensor.closed.Load() {
		return tensor.ErrTensorClosed
	}

	return nil
}

func (deviceTensor *DeviceTensor) Ready() <-chan struct{} {
	channel := make(chan struct{})
	close(channel)
	return channel
}

func (deviceTensor *DeviceTensor) RequiresGrad() bool {
	return deviceTensor.gradFlag.Load()
}

func (deviceTensor *DeviceTensor) SetRequiresGrad(yes bool) error {
	deviceTensor.gradFlag.Store(yes)
	return nil
}

func (deviceTensor *DeviceTensor) Grad() (tensor.Tensor, error) {
	return nil, tensor.ErrNoAutograd
}

func (deviceTensor *DeviceTensor) GradFn() tensor.GradFn {
	return nil
}

func (deviceTensor *DeviceTensor) Sync(ctx context.Context) error {
	if err := deviceTensor.WaitReady(); err != nil {
		return err
	}

	if deviceTensor.backend != nil {
		deviceTensor.backend.SyncDevice()
	}

	return ctx.Err()
}

func (deviceTensor *DeviceTensor) Close() error {
	if deviceTensor.workspaceView {
		return nil
	}

	if !deviceTensor.closed.CompareAndSwap(false, true) {
		return nil
	}

	if deviceTensor.ownsBuffer && deviceTensor.buffer != nil {
		C.metal_buffer_release(deviceTensor.buffer)
	}

	deviceTensor.buffer = nil
	deviceTensor.state.Store(uint32(tensor.StateClosed))
	return nil
}

func (deviceTensor *DeviceTensor) RawBytes() (dtype.DType, []byte, error) {
	if deviceTensor.backend == nil || deviceTensor.backend.bridge == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	return deviceTensor.backend.bridge.download(deviceTensor)
}

/*
DispatchPointer returns an unsafe.Pointer to the DeviceTensor struct
itself. The Metal compute host's resolveDeviceTensor unwraps this back
into a *DeviceTensor and then pulls the MTLBuffer out via the .buffer
field. This indirection is why Metal kernels can run through the same
puter/execution dispatcher that drives the CPU backend: every device
backend agrees that an unsafe.Pointer in the dispatch path is "whatever
the backend needs to find its resident storage", and each backend
implements its own resolve step.

Returning the struct pointer rather than the buffer handle directly
preserves the shape/dtype/state metadata the bridge uses for validation
(see resolveDeviceTensor: it checks .buffer != nil before unwrapping).
*/
func (deviceTensor *DeviceTensor) DispatchPointer() unsafe.Pointer {
	if deviceTensor == nil || deviceTensor.closed.Load() || deviceTensor.buffer == nil {
		return nil
	}

	return unsafe.Pointer(deviceTensor)
}

func (deviceTensor *DeviceTensor) Slice(start, length int) (tensor.Tensor, error) {
	if start != 0 || length < 0 || length > deviceTensor.Len() {
		return nil, tensor.ErrShapeMismatch
	}

	shape, err := tensor.NewShape([]int{length})

	if err != nil {
		return nil, err
	}

	byteCount, err := shape.Bytes(deviceTensor.elementFormat)

	if err != nil {
		return nil, err
	}

	if byteCount > deviceTensor.byteCount {
		return nil, tensor.ErrShapeMismatch
	}

	view := &DeviceTensor{
		backend:       deviceTensor.backend,
		shape:         shape,
		elementFormat: deviceTensor.elementFormat,
		buffer:        deviceTensor.buffer,
		byteCount:     byteCount,
		ownsBuffer:    false,
		workspaceView: deviceTensor.workspaceView,
	}

	view.state.Store(uint32(tensor.StateReady))

	return view, nil
}

func (deviceTensor *DeviceTensor) Reshape(dims []int) (tensor.Tensor, error) {
	shape, err := tensor.NewShape(dims)

	if err != nil {
		return nil, err
	}

	byteCount, err := shape.Bytes(deviceTensor.elementFormat)

	if err != nil {
		return nil, err
	}

	if byteCount != deviceTensor.byteCount {
		return nil, tensor.ErrShapeMismatch
	}

	view := &DeviceTensor{
		backend:       deviceTensor.backend,
		shape:         shape,
		elementFormat: deviceTensor.elementFormat,
		buffer:        deviceTensor.buffer,
		byteCount:     deviceTensor.byteCount,
		ownsBuffer:    false,
		workspaceView: deviceTensor.workspaceView,
	}

	view.state.Store(uint32(tensor.StateReady))

	return view, nil
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
func (deviceTensor *DeviceTensor) Int64Native() ([]int64, error) { return nil, tensor.ErrDTypeMismatch }
func (deviceTensor *DeviceTensor) Int32Native() ([]int32, error) { return nil, tensor.ErrDTypeMismatch }
func (deviceTensor *DeviceTensor) Int16Native() ([]int16, error) { return nil, tensor.ErrDTypeMismatch }
func (deviceTensor *DeviceTensor) Int8Native() ([]int8, error)   { return nil, tensor.ErrDTypeMismatch }
func (deviceTensor *DeviceTensor) Uint64Native() ([]uint64, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (deviceTensor *DeviceTensor) Uint32Native() ([]uint32, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (deviceTensor *DeviceTensor) Uint16Native() ([]uint16, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (deviceTensor *DeviceTensor) Uint8Native() ([]uint8, error) { return nil, tensor.ErrDTypeMismatch }
func (deviceTensor *DeviceTensor) BoolNative() (tensor.BitVector, error) {
	return tensor.BitVector{}, tensor.ErrDTypeMismatch
}
func (deviceTensor *DeviceTensor) Int4Native() (tensor.Int4Vector, error) {
	return tensor.Int4Vector{}, tensor.ErrDTypeMismatch
}

/*
NewBackend constructs a Metal backend on darwin+cgo builds.
*/
func NewBackend(ctx context.Context, workerPool *qpool.Q) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	backend := &Backend{
		ctx:    ctx,
		cancel: cancel,
		pool:   workerPool,
	}

	bridge, err := openMetalBridge(backend)

	if err != nil {
		cancel()
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge}
	backend.bindFamilies(computeHost)

	return backend, nil
}
