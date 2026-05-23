//go:build darwin && cgo

package metal

import (
	"github.com/theapemachine/puter/device"
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
	backend.Normalization = normalization.New(computeHost)
	backend.Norm = layernorm.New(computeHost)
	backend.RotaryEmbedding = rope.New(computeHost)
	backend.Hawkes = hawkes.New(computeHost)
	backend.Physics = physics.New(computeHost)
	backend.Causal = causal.New(computeHost)
	backend.Attention = attention.New(computeHost)
	backend.VSA = vsa.New(computeHost)
	backend.ActiveInference = active_inference.New(computeHost)
	backend.PredictiveCoding = predictive_coding.New(computeHost)
	backend.Dequantization = dequant.New(computeHost)
	backend.Quantization = quant.New(computeHost)
}

var _ device.Backend = (*Backend)(nil)
