package metal

import (
	"context"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal/activation"
	"github.com/theapemachine/puter/device/metal/active_inference"
	"github.com/theapemachine/puter/device/metal/attention"
	"github.com/theapemachine/puter/device/metal/causal"
	"github.com/theapemachine/puter/device/metal/convolution"
	"github.com/theapemachine/puter/device/metal/dequant"
	"github.com/theapemachine/puter/device/metal/dot"
	"github.com/theapemachine/puter/device/metal/dropout"
	"github.com/theapemachine/puter/device/metal/elementwise"
	"github.com/theapemachine/puter/device/metal/embedding"
	"github.com/theapemachine/puter/device/metal/hawkes"
	"github.com/theapemachine/puter/device/metal/layernorm"
	"github.com/theapemachine/puter/device/metal/losses"
	"github.com/theapemachine/puter/device/metal/matmul"
	"github.com/theapemachine/puter/device/metal/normalization"
	"github.com/theapemachine/puter/device/metal/physics"
	"github.com/theapemachine/puter/device/metal/pool"
	"github.com/theapemachine/puter/device/metal/predictive_coding"
	"github.com/theapemachine/puter/device/metal/quant"
	"github.com/theapemachine/puter/device/metal/reduction"
	"github.com/theapemachine/puter/device/metal/rope"
	"github.com/theapemachine/puter/device/metal/sampling"
	"github.com/theapemachine/puter/device/metal/vsa"
	"github.com/theapemachine/qpool"
)

/*
Backend is the Metal backend implementation.
*/
type Backend struct {
	ctx    context.Context
	cancel context.CancelFunc
	pool   *qpool.Q
	closed atomic.Bool
	bridge *metalBridge

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
	normalization.Normalization
	layernorm.Norm
	rope.RotaryEmbedding
	hawkes.Hawkes
	physics.Physics
	causal.Causal
	attention.Attention
	vsa.VSA
	active_inference.ActiveInference
	predictive_coding.PredictiveCoding
	dequant.Dequantization
	quant.Quantization
}

/*
Location reports Metal.
*/
func (backend *Backend) Location() tensor.Location {
	return tensor.Metal
}

/*
SupportedDTypes lists dtypes stored natively on Metal.
*/
func (backend *Backend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{
		dtype.Float32,
		dtype.Float16,
		dtype.BFloat16,
		dtype.Int32,
		dtype.Int8,
		dtype.Int4,
		dtype.Bool,
	}
}

/*
SupportedLayouts reports dense-only storage for Metal today.
*/
func (backend *Backend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

/*
Capabilities reports the Metal backend properties.
*/
func (backend *Backend) Capabilities() tensor.Capabilities {
	maxBytes := int64(0)

	if backend.bridge != nil {
		maxBytes = backend.bridge.recommendedMaxWorkingSet()
	}

	return tensor.Capabilities{
		MaxBytes:         maxBytes,
		SupportsAsync:    backend.bridge != nil,
		SupportsSparse:   false,
		SupportsAutograd: false,
		NativeAlignment:  256,
		NUMANodes:        1,
	}
}

/*
Upload copies host bytes into a shared MTLBuffer.
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
UploadAsync is synchronous on Metal shared memory today.
*/
func (backend *Backend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return backend.Upload(shape, sourceDType, bytesIn)
}

/*
UploadSparse is unsupported on Metal today.
*/
func (backend *Backend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

/*
Download copies device bytes back to the host.
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

/*
Close releases the Metal bridge.
*/
func (backend *Backend) Close() error {
	if !backend.closed.CompareAndSwap(false, true) {
		return nil
	}

	if backend.cancel != nil {
		backend.cancel()
	}

	if backend.bridge != nil {
		err := backend.bridge.close()
		backend.bridge = nil
		return err
	}

	return nil
}

var _ tensor.Backend = (*Backend)(nil)

func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.Activation.ReLU(dst, src, count, format)
}
