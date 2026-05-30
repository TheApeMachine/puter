package cpu

import (
	"github.com/theapemachine/puter/device/cpu/activation"
	"github.com/theapemachine/puter/device/cpu/active_inference"
	"github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/cpu/convolution"
	"github.com/theapemachine/puter/device/cpu/dequant"
	"github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cpu/dropout"
	"github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cpu/embedding"
	"github.com/theapemachine/puter/device/cpu/geometry"
	"github.com/theapemachine/puter/device/cpu/hawkes"
	"github.com/theapemachine/puter/device/cpu/interpretability"
	"github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/cpu/losses"
	"github.com/theapemachine/puter/device/cpu/masking"
	"github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cpu/normalization"
	"github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/cpu/pool"
	"github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/cpu/quant"
	"github.com/theapemachine/puter/device/cpu/reduction"
	"github.com/theapemachine/puter/device/cpu/rope"
	"github.com/theapemachine/puter/device/cpu/sampling"
	"github.com/theapemachine/puter/device/cpu/vsa"
)

func (backend *Backend) bindFamilies() {
	backend.Activation = activation.New()
	backend.Elementwise = elementwise.New()
	backend.Reduction = reduction.New()
	backend.Product = dot.New()
	backend.Gemm = matmul.New()
	backend.Pool = pool.New()
	backend.Convolution = convolution.New()
	backend.DropoutLayer = dropout.New()
	backend.Losses = losses.New()
	backend.Sampling = sampling.New()
	backend.Embedding = embedding.New()
	backend.Geometry = geometry.New()
	backend.Normalization = normalization.New()
	backend.Norm = layernorm.New()
	backend.RotaryEmbedding = rope.New()
	backend.Hawkes = hawkes.New()
	backend.Interpretability = interpretability.New()
	backend.Physics = physics.New()
	backend.Causal = causal.New()
	backend.Masking = masking.New()
	backend.Attention = attention.New()
	backend.VSA = vsa.New()
	backend.ActiveInference = active_inference.New()
	backend.PredictiveCoding = predictive_coding.New()
	backend.Dequantization = dequant.New()
	backend.Quantization = quant.New()
}
