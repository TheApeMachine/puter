//go:build cuda

package cuda

import (
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cuda/activation"
	"github.com/theapemachine/puter/device/cuda/active_inference"
	"github.com/theapemachine/puter/device/cuda/attention"
	"github.com/theapemachine/puter/device/cuda/causal"
	"github.com/theapemachine/puter/device/cuda/checkpoint"
	"github.com/theapemachine/puter/device/cuda/convolution"
	"github.com/theapemachine/puter/device/cuda/dequant"
	"github.com/theapemachine/puter/device/cuda/dot"
	"github.com/theapemachine/puter/device/cuda/dropout"
	"github.com/theapemachine/puter/device/cuda/elementwise"
	"github.com/theapemachine/puter/device/cuda/embedding"
	"github.com/theapemachine/puter/device/cuda/geometry"
	"github.com/theapemachine/puter/device/cuda/hawkes"
	"github.com/theapemachine/puter/device/cuda/interpretability"
	"github.com/theapemachine/puter/device/cuda/layernorm"
	"github.com/theapemachine/puter/device/cuda/losses"
	"github.com/theapemachine/puter/device/cuda/masking"
	"github.com/theapemachine/puter/device/cuda/math"
	"github.com/theapemachine/puter/device/cuda/matmul"
	"github.com/theapemachine/puter/device/cuda/model_editing"
	"github.com/theapemachine/puter/device/cuda/normalization"
	"github.com/theapemachine/puter/device/cuda/optimizer"
	"github.com/theapemachine/puter/device/cuda/peel"
	"github.com/theapemachine/puter/device/cuda/physics"
	"github.com/theapemachine/puter/device/cuda/pool"
	"github.com/theapemachine/puter/device/cuda/predictive_coding"
	"github.com/theapemachine/puter/device/cuda/quant"
	"github.com/theapemachine/puter/device/cuda/reduction"
	"github.com/theapemachine/puter/device/cuda/resonant"
	"github.com/theapemachine/puter/device/cuda/rope"
	"github.com/theapemachine/puter/device/cuda/sampling"
	"github.com/theapemachine/puter/device/cuda/shape"
	"github.com/theapemachine/puter/device/cuda/vsa"
)

var _ device.Backend = (*Backend)(nil)

/*
NewBackend constructs a CUDA backend. Returns ErrNeedsPlatformSetup
if no CUDA-capable device is present or the cgo toolchain is missing.
*/
func NewBackend() (*Backend, error) {
	backend := &Backend{}
	bridge, err := openCUDABridge(backend)

	if err != nil {
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge}
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
