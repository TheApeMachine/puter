//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
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

var _ device.Backend = (*Backend)(nil)

/*
NewBackend constructs an XLA backend. Returns ErrNeedsPlatformSetup
if the PJRT plugin cannot be opened on the current platform.
*/
func NewBackend() (*Backend, error) {
	backend := &Backend{
		workspace: NewWorkspace(),
		builder:   NewRuntimeBuilder(),
	}
	bridge, err := openXLABridge(backend)

	if err != nil {
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge, builder: backend.builder}
	backend.bindFamilies(computeHost)

	return backend, nil
}

func (backend *Backend) bindFamilies(computeHost *ComputeHost) {
	backend.Activation = activation.New(computeHost)
	backend.Elementwise = elementwise.New(computeHost)
	backend.Reduction = reduction.New(computeHost)
	backend.Product = dot.New(computeHost)
	backend.Gemm = matmul.New(computeHost)
	backend.Pool = pool.New(computeHost)
	backend.Convolution = convolution.New(computeHost)
	backend.DropoutLayer = dropout.New(computeHost)
	backend.Losses = losses.New(computeHost)
	backend.Sampling = sampling.New(computeHost)
	backend.Embedding = embedding.New(computeHost)
	backend.Geometry = geometry.New()
	backend.Normalization = normalization.New(computeHost)
	backend.Norm = layernorm.New(computeHost)
	backend.RotaryEmbedding = rope.New(computeHost)
	backend.Hawkes = hawkes.New(computeHost)
	backend.Physics = physics.New(computeHost)
	backend.Causal = causal.New(computeHost)
	backend.Masking = masking.New(computeHost)
	backend.Math = math.New()
	backend.Attention = attention.New(computeHost)
	backend.Checkpoint = checkpoint.New()
	backend.ModelEditing = model_editing.New()
	backend.Optimizer = optimizer.New()
	backend.Peel = peel.New()
	backend.Shape = shape.New()
	backend.VSA = vsa.New(computeHost)
	backend.Interpretability = interpretability.New()
	backend.ActiveInference = active_inference.New(computeHost)
	backend.PredictiveCoding = predictive_coding.New(computeHost)
	backend.Resonant = resonant.New()
	backend.Dequantization = dequant.New(computeHost)
	backend.Quantization = quant.New(computeHost)
}

func (backend *Backend) MatmulBiasGelu(
	out, left, right, bias unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
	host := &ComputeHost{bridge: backend.bridge, builder: backend.builder}
	host.MatmulBiasGeluLaunch(out, left, right, bias, rows, inner, cols, format)
}

func (backend *Backend) LayernormResidual(
	out, input, scale, bias, residual unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	host := &ComputeHost{bridge: backend.bridge, builder: backend.builder}
	host.LayernormResidualLaunch(out, input, scale, bias, residual, rows, lastDim, format)
}

func (backend *Backend) BuilderCacheMetrics() CacheMetrics {
	return backend.builder.CacheMetrics()
}

func (backend *Backend) uploadSparseCSR(
	shape tensor.Shape,
	valueDType dtype.DType,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	nnz := nnzFromCSRIndices(indices)

	expectedBytes, err := valueDType.BytesFor(nnz)

	if err != nil {
		return nil, err
	}

	if expectedBytes != len(values) {
		return nil, tensor.ErrShapeMismatch
	}

	valueShape, err := tensor.NewShape([]int{nnz})

	if err != nil {
		return nil, err
	}

	valueTensor, err := backend.Upload(valueShape, valueDType, values)

	if err != nil {
		return nil, err
	}

	valueDevice, ok := valueTensor.(*DeviceTensor)

	if !ok {
		_ = valueTensor.Close()
		return nil, tensor.ErrLayoutUnsupported
	}

	rowPtrSource := lookupCSRIndex(indices, "row_ptr")
	colIdxSource := lookupCSRIndex(indices, "col_idx")

	if rowPtrSource == nil || colIdxSource == nil {
		_ = valueDevice.Close()
		return nil, tensor.ErrShapeMismatch
	}

	rowPtrDevice, err := requireBackendDeviceTensor(backend, rowPtrSource)

	if err != nil {
		_ = valueDevice.Close()
		return nil, err
	}

	colIdxDevice, err := requireBackendDeviceTensor(backend, colIdxSource)

	if err != nil {
		_ = valueDevice.Close()
		return nil, err
	}

	return newDeviceSparseCSR(
		backend,
		shape,
		valueDType,
		valueDevice,
		rowPtrDevice,
		colIdxDevice,
		nnz,
	), nil
}
