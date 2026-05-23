//go:build xla

package xla

import (
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/xla/activation"
	"github.com/theapemachine/puter/device/xla/active_inference"
	"github.com/theapemachine/puter/device/xla/attention"
	"github.com/theapemachine/puter/device/xla/causal"
	"github.com/theapemachine/puter/device/xla/convolution"
	"github.com/theapemachine/puter/device/xla/dequant"
	"github.com/theapemachine/puter/device/xla/dot"
	"github.com/theapemachine/puter/device/xla/dropout"
	"github.com/theapemachine/puter/device/xla/elementwise"
	"github.com/theapemachine/puter/device/xla/embedding"
	"github.com/theapemachine/puter/device/xla/hawkes"
	"github.com/theapemachine/puter/device/xla/layernorm"
	"github.com/theapemachine/puter/device/xla/losses"
	"github.com/theapemachine/puter/device/xla/masking"
	"github.com/theapemachine/puter/device/xla/matmul"
	"github.com/theapemachine/puter/device/xla/normalization"
	"github.com/theapemachine/puter/device/xla/physics"
	"github.com/theapemachine/puter/device/xla/pool"
	"github.com/theapemachine/puter/device/xla/predictive_coding"
	"github.com/theapemachine/puter/device/xla/quant"
	"github.com/theapemachine/puter/device/xla/reduction"
	"github.com/theapemachine/puter/device/xla/rope"
	"github.com/theapemachine/puter/device/xla/sampling"
	"github.com/theapemachine/puter/device/xla/vsa"
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
	backend.Masking = masking.New(computeHost)
	backend.Attention = attention.New(computeHost)
	backend.VSA = vsa.New(computeHost)
	backend.ActiveInference = active_inference.New(computeHost)
	backend.PredictiveCoding = predictive_coding.New(computeHost)
	backend.Dequantization = dequant.New(computeHost)
	backend.Quantization = quant.New(computeHost)
}

var _ device.Backend = (*Backend)(nil)
