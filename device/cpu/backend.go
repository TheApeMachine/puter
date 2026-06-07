package cpu

import (
	"context"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/activation"
	"github.com/theapemachine/puter/device/cpu/active_inference"
	"github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/cpu/checkpoint"
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
	"github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cpu/model_editing"
	"github.com/theapemachine/puter/device/cpu/normalization"
	"github.com/theapemachine/puter/device/cpu/optimizer"
	"github.com/theapemachine/puter/device/cpu/peel"
	"github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/cpu/pool"
	"github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/cpu/quant"
	"github.com/theapemachine/puter/device/cpu/reduction"
	"github.com/theapemachine/puter/device/cpu/resonant"
	"github.com/theapemachine/puter/device/cpu/rope"
	"github.com/theapemachine/puter/device/cpu/sampling"
	"github.com/theapemachine/puter/device/cpu/shape"
	"github.com/theapemachine/puter/device/cpu/vsa"
	"github.com/theapemachine/qpool"
)

/*
Backend is the CPU device backend. Operation families are embedded
receivers; methods promote to satisfy device.Backend.
*/
type Backend struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
	pool   *qpool.Q[any]
	closed atomic.Bool

	workspaceMu     sync.Mutex
	workspaceBlocks []unsafe.Pointer

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
	interpretability.Interpretability
	physics.Physics
	causal.Causal
	masking.Masking
	math.Math
	checkpoint.Checkpoint
	model_editing.ModelEditing
	peel.Peel
	shape.Shape
	attention.Attention
	vsa.VSA
	active_inference.ActiveInference
	predictive_coding.PredictiveCoding
	resonant.Resonant
	optimizer.Stepper
	dequant.Dequantization
	quant.Quantization
}

var _ device.Backend = (*Backend)(nil)

/*
NewBackend constructs a CPU backend and wires embedded family receivers.
*/
func NewBackend(ctx context.Context, workerPool *qpool.Q[any]) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	backend := &Backend{
		ctx:    ctx,
		cancel: cancel,
		pool:   workerPool,
	}
	backend.bindFamilies()

	return backend, nil
}

/*
ReLU resolves the activation vs elementwise name collision in favor of
activation.
*/
func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.Activation.ReLU(dst, src, count, format)
}

/*
Close marks the backend closed and cancels its context.
*/
func (backend *Backend) Close() error {
	if !backend.closed.CompareAndSwap(false, true) {
		return nil
	}

	if backend.cancel != nil {
		backend.cancel()
	}

	backend.releaseWorkspace()

	return nil
}

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
	backend.Math = math.New()
	backend.Checkpoint = checkpoint.New()
	backend.ModelEditing = model_editing.New()
	backend.Peel = peel.New()
	backend.Shape = shape.New()
	backend.Attention = attention.New()
	backend.VSA = vsa.New()
	backend.ActiveInference = active_inference.New()
	backend.PredictiveCoding = predictive_coding.New()
	backend.Resonant = resonant.New()
	backend.Stepper = optimizer.NewStepper()
	backend.Dequantization = dequant.New()
	backend.Quantization = quant.New()
}
