/*
Package xla is the XLA device backend. Tensors are HLO buffers managed
through the XLA runtime's transfer manager. Upload is async by default;
Download blocks until the transfer event fires.

Per the spray-and-pray contract, this package compiles on every
platform; the XLA runtime bindings live in bridge_xla.go behind the
'xla' build tag and return ErrNeedsPlatformSetup elsewhere.
*/
package xla

import (
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/activation"
	"github.com/theapemachine/puter/device/xla/active_inference"
	"github.com/theapemachine/puter/device/xla/attention"
	"github.com/theapemachine/puter/device/xla/causal"
	"github.com/theapemachine/puter/device/xla/checkpoint"
	"github.com/theapemachine/puter/device/xla/convolution"
	"github.com/theapemachine/puter/device/xla/dequant"
	"github.com/theapemachine/puter/device/xla/dot"
	"github.com/theapemachine/puter/device/xla/dropout"
	"github.com/theapemachine/puter/device/xla/elementwise"
	"github.com/theapemachine/puter/device/xla/embedding"
	"github.com/theapemachine/puter/device/xla/geometry"
	"github.com/theapemachine/puter/device/xla/hawkes"
	"github.com/theapemachine/puter/device/xla/interpretability"
	"github.com/theapemachine/puter/device/xla/layernorm"
	"github.com/theapemachine/puter/device/xla/losses"
	"github.com/theapemachine/puter/device/xla/masking"
	"github.com/theapemachine/puter/device/xla/math"
	"github.com/theapemachine/puter/device/xla/matmul"
	"github.com/theapemachine/puter/device/xla/model_editing"
	"github.com/theapemachine/puter/device/xla/normalization"
	"github.com/theapemachine/puter/device/xla/optimizer"
	"github.com/theapemachine/puter/device/xla/peel"
	"github.com/theapemachine/puter/device/xla/physics"
	"github.com/theapemachine/puter/device/xla/pool"
	"github.com/theapemachine/puter/device/xla/predictive_coding"
	"github.com/theapemachine/puter/device/xla/quant"
	"github.com/theapemachine/puter/device/xla/reduction"
	"github.com/theapemachine/puter/device/xla/resonant"
	"github.com/theapemachine/puter/device/xla/rope"
	"github.com/theapemachine/puter/device/xla/sampling"
	"github.com/theapemachine/puter/device/xla/shape"
	"github.com/theapemachine/puter/device/xla/vsa"
)

/*
Backend is the XLA Backend implementation.
*/
type Backend struct {
	closed    atomic.Bool
	bridge    *xlaBridge
	workspace *Workspace
	builder   *Builder

	activation.Activation
	elementwise.Elementwise
	reduction.Reduction
	dot.Product
	matmul.Gemm
	pool.Pool
	convolution.Convolution
	dropout.DropoutLayer
	losses.Losses
	sampling.Sampling
	embedding.Embedding
	geometry.Geometry
	normalization.Normalization
	layernorm.Norm
	rope.RotaryEmbedding
	hawkes.Hawkes
	physics.Physics
	causal.Causal
	masking.Masking
	attention.Attention
	math.Math
	checkpoint.Checkpoint
	model_editing.ModelEditing
	optimizer.Optimizer
	peel.Peel
	shape.Shape
	interpretability.Interpretability
	vsa.VSA
	active_inference.ActiveInference
	predictive_coding.PredictiveCoding
	resonant.Resonant
	dequant.Dequantization
	quant.Quantization
}

/*
Close releases the bridge.
*/
func (backend *Backend) Close() error {
	if !backend.closed.CompareAndSwap(false, true) {
		return nil
	}

	if backend.workspace != nil {
		backend.releaseWorkspace()
		backend.workspace.Close()
		backend.workspace = nil
	}

	if backend.bridge != nil {
		err := backend.bridge.close()
		backend.bridge = nil
		return err
	}

	return nil
}

/*
Location reports XLA.
*/
func (backend *Backend) Location() tensor.Location {
	return tensor.XLA
}

/*
SupportedDTypes returns the XLA-native dtype set, which is the
broadest of the device backends: every IEEE 754 float plus the
quantized integer family.
*/
func (backend *Backend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{
		dtype.Float64,
		dtype.Float32,
		dtype.Float16,
		dtype.BFloat16,
		dtype.Float8E4M3,
		dtype.Float8E5M2,
		dtype.Int64,
		dtype.Int32,
		dtype.Int16,
		dtype.Int8,
		dtype.Uint64,
		dtype.Uint32,
		dtype.Uint16,
		dtype.Uint8,
		dtype.Bool,
	}
}

/*
SupportedLayouts includes dense and CSR sparse storage.
*/
func (backend *Backend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense, tensor.LayoutSparseCSR}
}

/*
Capabilities returns the XLA backend's properties. MaxBytes depends on
the platform (TPU pod vs. XLA-GPU); the bridge fills it in on open.
*/
func (backend *Backend) Capabilities() tensor.Capabilities {
	maxBytes := int64(0)

	if backend.bridge != nil {
		maxBytes = backend.bridge.devicePoolBytes()
	}

	return tensor.Capabilities{
		MaxBytes:         maxBytes,
		SupportsAsync:    backend.bridge != nil,
		SupportsSparse:   backend.bridge != nil,
		SupportsAutograd: false,
		NativeAlignment:  128,
		NUMANodes:        1,
	}
}

/*
Upload routes through the XLA transfer manager. Synchronous variant.
*/
func (backend *Backend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.upload(shape, sourceDType, bytesIn)
}

/*
UploadAsync issues an XLA async transfer. The returned tensor is in
StatePending until the transfer event fires.
*/
func (backend *Backend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.uploadAsync(shape, sourceDType, bytesIn)
}

/*
UploadSparse stores CSR sparse tensors as XLA-resident value and index buffers.
*/
func (backend *Backend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	if layout != tensor.LayoutSparseCSR {
		return nil, tensor.ErrLayoutUnsupported
	}

	return backend.uploadSparseCSR(shape, valueDType, values, indices)
}

/*
Download materializes a host copy of the tensor's storage.
*/
func (backend *Backend) Download(input tensor.Tensor) (dtype.DType, []byte, error) {
	if backend.closed.Load() {
		return dtype.Invalid, nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.download(input)
}

func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.Activation.ReLU(dst, src, count, format)
}

var _ tensor.Backend = (*Backend)(nil)
