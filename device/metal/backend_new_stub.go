//go:build !darwin || !cgo

package metal

import (
	"context"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal/activation"
	"github.com/theapemachine/puter/device/metal/active_inference"
	"github.com/theapemachine/puter/device/metal/attention"
	"github.com/theapemachine/puter/device/metal/causal"
	"github.com/theapemachine/puter/device/metal/checkpoint"
	"github.com/theapemachine/puter/device/metal/convolution"
	"github.com/theapemachine/puter/device/metal/dequant"
	"github.com/theapemachine/puter/device/metal/dot"
	"github.com/theapemachine/puter/device/metal/dropout"
	"github.com/theapemachine/puter/device/metal/elementwise"
	"github.com/theapemachine/puter/device/metal/embedding"
	"github.com/theapemachine/puter/device/metal/geometry"
	"github.com/theapemachine/puter/device/metal/hawkes"
	"github.com/theapemachine/puter/device/metal/interpretability"
	"github.com/theapemachine/puter/device/metal/layernorm"
	"github.com/theapemachine/puter/device/metal/losses"
	"github.com/theapemachine/puter/device/metal/masking"
	"github.com/theapemachine/puter/device/metal/math"
	"github.com/theapemachine/puter/device/metal/matmul"
	"github.com/theapemachine/puter/device/metal/model_editing"
	"github.com/theapemachine/puter/device/metal/normalization"
	"github.com/theapemachine/puter/device/metal/optimizer"
	"github.com/theapemachine/puter/device/metal/peel"
	"github.com/theapemachine/puter/device/metal/physics"
	"github.com/theapemachine/puter/device/metal/pool"
	"github.com/theapemachine/puter/device/metal/predictive_coding"
	"github.com/theapemachine/puter/device/metal/quant"
	"github.com/theapemachine/puter/device/metal/reduction"
	"github.com/theapemachine/puter/device/metal/resonant"
	"github.com/theapemachine/puter/device/metal/rope"
	"github.com/theapemachine/puter/device/metal/sampling"
	"github.com/theapemachine/puter/device/metal/shape"
	"github.com/theapemachine/puter/device/metal/vsa"
	"github.com/theapemachine/qpool"
)

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
	backend.Interpretability = interpretability.New(computeHost)
	backend.Physics = physics.New(computeHost)
	backend.Causal = causal.New(computeHost)
	backend.Masking = masking.New(computeHost)
	backend.Math = math.New(computeHost)
	backend.Attention = attention.New(computeHost)
	backend.Checkpoint = checkpoint.New(computeHost)
	backend.ModelEditing = model_editing.New(computeHost)
	backend.Optimizer = optimizer.New(computeHost)
	backend.Peel = peel.New()
	backend.Shape = shape.New(computeHost)
	backend.VSA = vsa.New(computeHost)
	backend.ActiveInference = active_inference.New(computeHost)
	backend.PredictiveCoding = predictive_coding.New(computeHost)
	backend.Resonant = resonant.New(computeHost)
	backend.Dequantization = dequant.New(computeHost)
	backend.Quantization = quant.New(computeHost)
}

type metalBridge struct {
	backend *Backend
}

func openMetalBridge(backend *Backend) (*metalBridge, error) {
	return &metalBridge{backend: backend}, nil
}

func (backend *Backend) BeginBatch() error {
	return nil
}

func (backend *Backend) EndBatch() error {
	return nil
}

func (bridge *metalBridge) recommendedMaxWorkingSet() int64 {
	return 0
}

func (bridge *metalBridge) close() error {
	return nil
}

type ComputeHost struct {
	bridge *metalBridge
}

func (host *ComputeHost) NeedsPlatform() {
	panic("metal: platform unavailable")
}

func (host *ComputeHost) unavailable() {
	panic("metal: dispatch not implemented")
}

func (bridge *metalBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

/*
NewBackend constructs a Metal backend on non-darwin builds.
*/
func NewBackend(ctx context.Context, workerPool *qpool.Q[any]) (*Backend, error) {
	backend := &Backend{
		pool: workerPool,
	}

	bridge, err := openMetalBridge(backend)

	if err != nil {
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge}
	backend.bindFamilies(computeHost)

	return backend, nil
}
