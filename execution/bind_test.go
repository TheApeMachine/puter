package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

type recordingDevice struct {
	addCalls           []recordedAddCall
	timestepCalls      []recordedTimestepCall
	conv2DCalls        []recordedConv2DCall
	matmulCalls        []recordedMatmulCall
	rmsNormCalls       []recordedRMSNormCall
	adaptiveNormCalls  []recordedAdaptiveRMSNormCall
	modulatedNormCalls []recordedModulatedLayerNormCall
	groupNormCalls     []recordedGroupNormCall
	batchDenormCalls   []recordedBatchNormDenormCall
	swiGLUTensorCalls  []recordedSwiGLUTensorCall
	ropeCalls          []recordedRoPECall
	multiAxisRoPECalls []recordedMultiAxisRoPECall
	multiHeadCalls     []recordedMultiHeadAttentionCall
}

type recordedAddCall struct {
	dst    unsafe.Pointer
	left   unsafe.Pointer
	right  unsafe.Pointer
	count  int
	format dtype.DType
}

type recordedSwiGLUTensorCall struct {
	dst    unsafe.Pointer
	gate   unsafe.Pointer
	up     unsafe.Pointer
	count  int
	format dtype.DType
}

type recordedTimestepCall struct {
	config    device.TimestepEmbeddingConfig
	timesteps unsafe.Pointer
	output    unsafe.Pointer
	count     int
	dim       int
	format    dtype.DType
}

type recordedConv2DCall struct {
	config      device.Conv2DConfig
	input       unsafe.Pointer
	weight      unsafe.Pointer
	bias        unsafe.Pointer
	output      unsafe.Pointer
	batch       int
	inChannels  int
	inHeight    int
	inWidth     int
	outChannels int
	kernelH     int
	kernelW     int
	outHeight   int
	outWidth    int
	format      dtype.DType
}

type recordedMatmulCall struct {
	output unsafe.Pointer
	left   unsafe.Pointer
	right  unsafe.Pointer
	rows   int
	inner  int
	cols   int
	format dtype.DType
}

type recordedRMSNormCall struct {
	config  device.RMSNormConfig
	input   unsafe.Pointer
	scale   unsafe.Pointer
	output  unsafe.Pointer
	rows    int
	lastDim int
	format  dtype.DType
}

type recordedAdaptiveRMSNormCall struct {
	config         device.RMSNormConfig
	input          unsafe.Pointer
	modulation     unsafe.Pointer
	output         unsafe.Pointer
	rows           int
	lastDim        int
	rowsPerBatch   int
	modulationCols int
	format         dtype.DType
}

type recordedModulatedLayerNormCall struct {
	config         device.ModulatedLayerNormConfig
	input          unsafe.Pointer
	modulation     unsafe.Pointer
	output         unsafe.Pointer
	rows           int
	lastDim        int
	rowsPerBatch   int
	modulationCols int
	format         dtype.DType
}

type recordedGroupNormCall struct {
	config   device.GroupNormConfig
	input    unsafe.Pointer
	scale    unsafe.Pointer
	bias     unsafe.Pointer
	output   unsafe.Pointer
	batch    int
	channels int
	spatial  int
	format   dtype.DType
}

type recordedBatchNormDenormCall struct {
	input    unsafe.Pointer
	mean     unsafe.Pointer
	variance unsafe.Pointer
	output   unsafe.Pointer
	batch    int
	channels int
	spatial  int
	format   dtype.DType
}

type recordedRoPECall struct {
	config   device.RoPEConfig
	input    unsafe.Pointer
	output   unsafe.Pointer
	seqLen   int
	numHeads int
	headDim  int
	format   dtype.DType
}

type recordedMultiAxisRoPECall struct {
	config   device.MultiAxisRoPEConfig
	input    unsafe.Pointer
	output   unsafe.Pointer
	batch    int
	seqLen   int
	numHeads int
	headDim  int
	format   dtype.DType
}

type recordedMultiHeadAttentionCall struct {
	config device.MultiHeadAttentionConfig
	query  unsafe.Pointer
	key    unsafe.Pointer
	value  unsafe.Pointer
	output unsafe.Pointer
	seqQ   int
	seqK   int
	format dtype.DType
}

func (recorder *recordingDevice) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	recorder.addCalls = append(recorder.addCalls, recordedAddCall{
		dst:    dst,
		left:   left,
		right:  right,
		count:  count,
		format: format,
	})
}

func (recorder *recordingDevice) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	recorder.swiGLUTensorCalls = append(recorder.swiGLUTensorCalls, recordedSwiGLUTensorCall{
		dst:    dst,
		gate:   gate,
		up:     up,
		count:  count,
		format: format,
	})
}

func (recordingDevice) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	panic("recordingDevice.Lookup invoked")
}

func (recorder *recordingDevice) TimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	recorder.timestepCalls = append(recorder.timestepCalls, recordedTimestepCall{
		config:    config,
		timesteps: timesteps,
		output:    output,
		count:     count,
		dim:       dim,
		format:    format,
	})
}

func (recorder *recordingDevice) RMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	recorder.rmsNormCalls = append(recorder.rmsNormCalls, recordedRMSNormCall{
		config:  config,
		input:   input,
		scale:   scale,
		output:  output,
		rows:    rows,
		lastDim: lastDim,
		format:  format,
	})
}

func (recorder *recordingDevice) AdaptiveRMSNorm(
	config device.RMSNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	recorder.adaptiveNormCalls = append(recorder.adaptiveNormCalls, recordedAdaptiveRMSNormCall{
		config:         config,
		input:          input,
		modulation:     modulation,
		output:         output,
		rows:           rows,
		lastDim:        lastDim,
		rowsPerBatch:   rowsPerBatch,
		modulationCols: modulationCols,
		format:         format,
	})
}

func (recordingDevice) LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("recordingDevice.LayerNorm invoked")
}

func (recorder *recordingDevice) ModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	recorder.modulatedNormCalls = append(recorder.modulatedNormCalls, recordedModulatedLayerNormCall{
		config:         config,
		input:          input,
		modulation:     modulation,
		output:         output,
		rows:           rows,
		lastDim:        lastDim,
		rowsPerBatch:   rowsPerBatch,
		modulationCols: modulationCols,
		format:         format,
	})
}

func (recorder *recordingDevice) BatchNormDenorm(
	input, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	recorder.batchDenormCalls = append(recorder.batchDenormCalls, recordedBatchNormDenormCall{
		input:    input,
		mean:     mean,
		variance: variance,
		output:   output,
		batch:    batch,
		channels: channels,
		spatial:  spatial,
		format:   format,
	})
}

func (recorder *recordingDevice) GroupNorm(
	config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	recorder.groupNormCalls = append(recorder.groupNormCalls, recordedGroupNormCall{
		config:   config,
		input:    input,
		scale:    scale,
		bias:     bias,
		output:   output,
		batch:    batch,
		channels: channels,
		spatial:  spatial,
		format:   format,
	})
}

func (recorder *recordingDevice) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	recorder.matmulCalls = append(recorder.matmulCalls, recordedMatmulCall{
		output: out,
		left:   left,
		right:  right,
		rows:   rows,
		inner:  inner,
		cols:   cols,
		format: format,
	})
}

func (recorder *recordingDevice) Conv2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	recorder.conv2DCalls = append(recorder.conv2DCalls, recordedConv2DCall{
		config:      config,
		input:       input,
		weight:      weight,
		bias:        bias,
		output:      output,
		batch:       batch,
		inChannels:  inChannels,
		inHeight:    inHeight,
		inWidth:     inWidth,
		outChannels: outChannels,
		kernelH:     kernelHeight,
		kernelW:     kernelWidth,
		outHeight:   outHeight,
		outWidth:    outWidth,
		format:      format,
	})
}

func (recordingDevice) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Sub invoked")
}

func (recordingDevice) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Mul invoked")
}

func (recordingDevice) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Div invoked")
}

func (recordingDevice) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.ReLU invoked")
}

func (recordingDevice) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Sigmoid invoked")
}

func (recordingDevice) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Tanh invoked")
}

func (recordingDevice) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Gelu invoked")
}

func (recordingDevice) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.Silu invoked")
}

func (recordingDevice) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	panic("recordingDevice.SwiGLU invoked")
}

func (recorder *recordingDevice) RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	recorder.ropeCalls = append(recorder.ropeCalls, recordedRoPECall{
		config:   config,
		input:    input,
		output:   output,
		seqLen:   seqLen,
		numHeads: numHeads,
		headDim:  headDim,
		format:   format,
	})
}

func (recorder *recordingDevice) MultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	recorder.multiAxisRoPECalls = append(recorder.multiAxisRoPECalls, recordedMultiAxisRoPECall{
		config:   config,
		input:    input,
		output:   output,
		batch:    batch,
		seqLen:   seqLen,
		numHeads: numHeads,
		headDim:  headDim,
		format:   format,
	})
}

func (recorder *recordingDevice) MultiHeadAttention(
	config device.MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	recorder.multiHeadCalls = append(recorder.multiHeadCalls, recordedMultiHeadAttentionCall{
		config: config,
		query:  query,
		key:    key,
		value:  value,
		output: output,
		seqQ:   seqQ,
		seqK:   seqK,
		format: format,
	})
}

func (recordingDevice) ResonantUpdateForward(
	x, y, vr, vi, diag unsafe.Pointer,
	xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	panic("recordingDevice.ResonantUpdateForward invoked")
}

func (recordingDevice) ResonantUpdateBackward(
	gradXOut, gradYOut unsafe.Pointer,
	x, y, diag, a, b, invR unsafe.Pointer,
	gradX, gradY, gradVR, gradVI unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	panic("recordingDevice.ResonantUpdateBackward invoked")
}

func TestRunBoundNodeUsesOperationYAML(t *testing.T) {
	convey.Convey("Given math.add is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		leftTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})
		rightTensor := uploadFloatSlice(t, memory, []float32{10, 20, 30, 40})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", leftTensor)
		dispatcher.values.set("y", rightTensor)

		node := &ast.GraphNode{
			ID:     "added",
			Op:     "math.add",
			Inputs: []string{"x", "y"},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one Add device call", func() {
			convey.So(len(recorder.addCalls), convey.ShouldEqual, 1)
			convey.So(recorder.addCalls[0].count, convey.ShouldEqual, 4)
			convey.So(recorder.addCalls[0].format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor is registered under the node ID", func() {
			stored, err := dispatcher.values.tensor("added")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{4})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesTimestepEmbeddingBind(t *testing.T) {
	convey.Convey("Given embedding.timestep is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		timestepTensor := uploadFloatSlice(t, memory, []float32{250})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("timestep", timestepTensor)

		node := &ast.GraphNode{
			ID:     "time_guidance_embed.time_proj",
			Op:     "embedding.timestep",
			Inputs: []string{"timestep"},
			Attributes: map[string]any{
				"dim":                  256,
				"flip_sin_to_cos":      true,
				"downscale_freq_shift": 0,
				"max_period":           10000,
				"timestep_divisor":     1000,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one TimestepEmbedding device call", func() {
			convey.So(len(recorder.timestepCalls), convey.ShouldEqual, 1)
			convey.So(recorder.timestepCalls[0].config.MaxPeriod, convey.ShouldEqual, float32(10000))
			convey.So(recorder.timestepCalls[0].config.TimestepDivisor, convey.ShouldEqual, float32(1000))
			convey.So(recorder.timestepCalls[0].config.FlipSinToCos, convey.ShouldBeTrue)
			convey.So(recorder.timestepCalls[0].count, convey.ShouldEqual, 1)
			convey.So(recorder.timestepCalls[0].dim, convey.ShouldEqual, 256)
			convey.So(recorder.timestepCalls[0].format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape is [batch, dim]", func() {
			stored, err := dispatcher.values.tensor("time_guidance_embed.time_proj")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 256})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesModulatedLayerNormBind(t *testing.T) {
	convey.Convey("Given math.modulated_layernorm is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*3*4), []int{2, 3, 4})
		modulationTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*24), []int{2, 24})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", inputTensor)
		dispatcher.values.set("modulation", modulationTensor)

		node := &ast.GraphNode{
			ID:     "modulated",
			Op:     "math.modulated_layernorm",
			Inputs: []string{"x", "modulation"},
			Attributes: map[string]any{
				"eps": 1e-6,
				"set": 1,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one ModulatedLayerNorm device call", func() {
			convey.So(len(recorder.modulatedNormCalls), convey.ShouldEqual, 1)

			call := recorder.modulatedNormCalls[0]
			convey.So(call.config.Epsilon, convey.ShouldEqual, float64(float32(1e-6)))
			convey.So(call.config.Set, convey.ShouldEqual, 1)
			convey.So(call.rows, convey.ShouldEqual, 6)
			convey.So(call.lastDim, convey.ShouldEqual, 4)
			convey.So(call.rowsPerBatch, convey.ShouldEqual, 3)
			convey.So(call.modulationCols, convey.ShouldEqual, 24)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the input", func() {
			stored, err := dispatcher.values.tensor("modulated")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{2, 3, 4})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesAdaptiveRMSNormBind(t *testing.T) {
	convey.Convey("Given math.adaptive_rmsnorm is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*3*4), []int{2, 3, 4})
		modulationTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*8), []int{2, 8})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", inputTensor)
		dispatcher.values.set("modulation", modulationTensor)

		node := &ast.GraphNode{
			ID:     "adaptive",
			Op:     "math.adaptive_rmsnorm",
			Inputs: []string{"x", "modulation"},
			Attributes: map[string]any{
				"eps": 1e-6,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one AdaptiveRMSNorm device call", func() {
			convey.So(len(recorder.adaptiveNormCalls), convey.ShouldEqual, 1)

			call := recorder.adaptiveNormCalls[0]
			convey.So(call.config.Epsilon, convey.ShouldEqual, float64(float32(1e-6)))
			convey.So(call.rows, convey.ShouldEqual, 6)
			convey.So(call.lastDim, convey.ShouldEqual, 4)
			convey.So(call.rowsPerBatch, convey.ShouldEqual, 3)
			convey.So(call.modulationCols, convey.ShouldEqual, 8)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the input", func() {
			stored, err := dispatcher.values.tensor("adaptive")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{2, 3, 4})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesBatchNormDenormBind(t *testing.T) {
	convey.Convey("Given math.batchnorm_denorm is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*3*5*7), []int{2, 3, 5, 7})
		meanTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3})
		varianceTensor := uploadFloatSlice(t, memory, []float32{0.25, 1, 4})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = mapWeightStore{
			"bn.running_mean": meanTensor,
			"bn.running_var":  varianceTensor,
		}
		dispatcher.values.set("x", inputTensor)

		node := &ast.GraphNode{
			ID:     "bn",
			Op:     "math.batchnorm_denorm",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"channels": 3,
				"mean":     "bn.running_mean",
				"variance": "bn.running_var",
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one BatchNormDenorm device call", func() {
			convey.So(len(recorder.batchDenormCalls), convey.ShouldEqual, 1)

			call := recorder.batchDenormCalls[0]
			convey.So(call.batch, convey.ShouldEqual, 2)
			convey.So(call.channels, convey.ShouldEqual, 3)
			convey.So(call.spatial, convey.ShouldEqual, 35)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the input", func() {
			stored, err := dispatcher.values.tensor("bn")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{2, 3, 5, 7})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesGroupNormBind(t *testing.T) {
	convey.Convey("Given math.groupnorm is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*4*5*7), []int{2, 4, 5, 7})
		scaleTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})
		biasTensor := uploadFloatSlice(t, memory, []float32{5, 6, 7, 8})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = mapWeightStore{
			"decoder.norm.weight": scaleTensor,
			"decoder.norm.bias":   biasTensor,
		}
		dispatcher.values.set("x", inputTensor)

		node := &ast.GraphNode{
			ID:     "norm",
			Op:     "math.groupnorm",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"groups": 2,
			},
			Weights: &ast.BoundWeight{
				TensorName: "decoder.norm.weight",
				BiasName:   "decoder.norm.bias",
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one GroupNorm device call", func() {
			convey.So(len(recorder.groupNormCalls), convey.ShouldEqual, 1)

			call := recorder.groupNormCalls[0]
			convey.So(call.config.Groups, convey.ShouldEqual, 2)
			convey.So(call.batch, convey.ShouldEqual, 2)
			convey.So(call.channels, convey.ShouldEqual, 4)
			convey.So(call.spatial, convey.ShouldEqual, 35)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the input", func() {
			stored, err := dispatcher.values.tensor("norm")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{2, 4, 5, 7})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesConv2DBind(t *testing.T) {
	convey.Convey("Given convolution.conv2d is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 1*3*5*7), []int{1, 3, 5, 7})
		weightTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 4*3*3*3), []int{4, 3, 3, 3})
		biasTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = mapWeightStore{
			"decoder.conv.weight": weightTensor,
			"decoder.conv.bias":   biasTensor,
		}
		dispatcher.values.set("x", inputTensor)

		node := &ast.GraphNode{
			ID:     "conv",
			Op:     "convolution.conv2d",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"in_channels":  3,
				"out_channels": 4,
				"kernel_h":     3,
				"kernel_w":     3,
				"stride_h":     1,
				"stride_w":     1,
				"pad_h":        1,
				"pad_w":        1,
				"dil_h":        1,
				"dil_w":        1,
			},
			Weights: &ast.BoundWeight{
				TensorName: "decoder.conv.weight",
				BiasName:   "decoder.conv.bias",
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one Conv2D device call", func() {
			convey.So(len(recorder.conv2DCalls), convey.ShouldEqual, 1)

			call := recorder.conv2DCalls[0]
			convey.So(call.config.StrideH, convey.ShouldEqual, 1)
			convey.So(call.config.PaddingH, convey.ShouldEqual, 1)
			convey.So(call.config.DilationH, convey.ShouldEqual, 1)
			convey.So(call.batch, convey.ShouldEqual, 1)
			convey.So(call.inChannels, convey.ShouldEqual, 3)
			convey.So(call.inHeight, convey.ShouldEqual, 5)
			convey.So(call.inWidth, convey.ShouldEqual, 7)
			convey.So(call.outChannels, convey.ShouldEqual, 4)
			convey.So(call.kernelH, convey.ShouldEqual, 3)
			convey.So(call.kernelW, convey.ShouldEqual, 3)
			convey.So(call.outHeight, convey.ShouldEqual, 5)
			convey.So(call.outWidth, convey.ShouldEqual, 7)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the convolution formula", func() {
			stored, err := dispatcher.values.tensor("conv")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 4, 5, 7})
		})
	})
}

func TestRunBoundNodeDerivesBiasFromWeightName(t *testing.T) {
	convey.Convey("Given a weighted convolution node with the default weight name", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 1*3*5*7), []int{1, 3, 5, 7})
		weightTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 4*3*3*3), []int{4, 3, 3, 3})
		biasTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = mapWeightStore{
			"decoder.conv.weight": weightTensor,
			"decoder.conv.bias":   biasTensor,
		}
		dispatcher.values.set("x", inputTensor)

		node := &ast.GraphNode{
			ID:     "conv",
			Op:     "convolution.conv2d",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"in_channels":  3,
				"out_channels": 4,
				"kernel_h":     3,
				"kernel_w":     3,
				"stride_h":     1,
				"stride_w":     1,
				"pad_h":        1,
				"pad_w":        1,
				"dil_h":        1,
				"dil_w":        1,
			},
			Weights: &ast.BoundWeight{
				TensorName: "decoder.conv.weight",
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router resolved the sibling bias tensor", func() {
			convey.So(len(recorder.conv2DCalls), convey.ShouldEqual, 1)
			convey.So(recorder.conv2DCalls[0].bias, convey.ShouldNotBeNil)
		})
	})
}

func BenchmarkRunBoundNodeConv2DBind(benchmark *testing.B) {
	memory := tensor.NewHostBackend()
	defer memory.Close()

	inputTensor := uploadFloatSliceWithShape(benchmark, memory, make([]float32, 1*3*8*8), []int{1, 3, 8, 8})
	weightTensor := uploadFloatSliceWithShape(benchmark, memory, make([]float32, 4*3*3*3), []int{4, 3, 3, 3})
	biasTensor := uploadFloatSlice(benchmark, memory, []float32{1, 2, 3, 4})

	recorder := &recordingDevice{}
	dispatcher := newTestDispatcher(recorder, memory)
	dispatcher.weights = mapWeightStore{
		"decoder.conv.weight": weightTensor,
		"decoder.conv.bias":   biasTensor,
	}
	dispatcher.values.set("x", inputTensor)

	node := &ast.GraphNode{
		ID:     "conv",
		Op:     "convolution.conv2d",
		Inputs: []string{"x"},
		Attributes: map[string]any{
			"in_channels":  3,
			"out_channels": 4,
			"kernel_h":     3,
			"kernel_w":     3,
			"stride_h":     1,
			"stride_w":     1,
			"pad_h":        1,
			"pad_w":        1,
			"dil_h":        1,
			"dil_w":        1,
		},
		Weights: &ast.BoundWeight{
			TensorName: "decoder.conv.weight",
			BiasName:   "decoder.conv.bias",
		},
	}

	for benchmark.Loop() {
		if err := dispatcher.runNode(node); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func TestRunBoundNodeUsesLiveViewOfWorkspaceOutput(t *testing.T) {
	convey.Convey("Given a planner workspace output larger than the live bind shape", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		leftTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3})
		rightTensor := uploadFloatSlice(t, memory, []float32{10, 20, 30})
		workspaceOutput := uploadFloatSlice(t, memory, []float32{0, 0, 0, 0, 0, 0, 0, 0})

		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.workspaces = &WorkspaceMap{
			outputs: map[string]map[string]tensor.Tensor{
				"test": {"added": workspaceOutput},
			},
		}
		dispatcher.values.set("x", leftTensor)
		dispatcher.values.set("y", rightTensor)

		node := &ast.GraphNode{
			ID:     "added",
			Op:     "math.add",
			Inputs: []string{"x", "y"},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The device call uses the live element count", func() {
			convey.So(len(recorder.addCalls), convey.ShouldEqual, 1)
			convey.So(recorder.addCalls[0].count, convey.ShouldEqual, 3)
		})

		convey.Convey("The value table stores a live-shape view", func() {
			stored, err := dispatcher.values.tensor("added")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{3})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeResolvesRoPEConfig(t *testing.T) {
	convey.Convey("Given positional.rope carries Llama 3 half-mode config", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, make([]float32, 1*2*2*4), []int{1, 2, 2, 4})
		position := uploadInt32Slice(t, memory, []int32{7})
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", input)
		dispatcher.values.set("position", position)

		node := &ast.GraphNode{
			ID:     "rope",
			Op:     "positional.rope",
			Inputs: []string{"x", "position"},
			Attributes: map[string]any{
				"base":                  500000.0,
				"mode":                  "half",
				"rope_type":             "llama3",
				"rope_factor":           32.0,
				"rope_low_freq_factor":  1.0,
				"rope_high_freq_factor": 4.0,
				"rope_original_context": 8192,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router passes the full RoPEConfig to the device", func() {
			convey.So(len(recorder.ropeCalls), convey.ShouldEqual, 1)
			convey.So(recorder.ropeCalls[0].seqLen, convey.ShouldEqual, 2)
			convey.So(recorder.ropeCalls[0].numHeads, convey.ShouldEqual, 2)
			convey.So(recorder.ropeCalls[0].headDim, convey.ShouldEqual, 4)
			convey.So(recorder.ropeCalls[0].format, convey.ShouldEqual, dtype.Float32)
			convey.So(recorder.ropeCalls[0].config.BaseFreq, convey.ShouldEqual, 500000.0)
			convey.So(recorder.ropeCalls[0].config.StartPosition, convey.ShouldEqual, 7)
			convey.So(recorder.ropeCalls[0].config.Mode, convey.ShouldEqual, device.RoPEModeHalf)
			convey.So(recorder.ropeCalls[0].config.Scaling, convey.ShouldEqual, device.RoPEScalingLlama3)
			convey.So(recorder.ropeCalls[0].config.ScalingFactor, convey.ShouldEqual, 32.0)
			convey.So(recorder.ropeCalls[0].config.LowFreqFactor, convey.ShouldEqual, 1.0)
			convey.So(recorder.ropeCalls[0].config.HighFreqFactor, convey.ShouldEqual, 4.0)
			convey.So(recorder.ropeCalls[0].config.OriginalContext, convey.ShouldEqual, 8192)
		})
	})
}

func TestRunBoundNodeUsesMultiAxisRoPEBind(t *testing.T) {
	convey.Convey("Given positional.multi_axis_rope is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, make([]float32, 1*8*2*8), []int{1, 8, 2, 8})
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "rope",
			Op:     "positional.multi_axis_rope",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"base":           2000.0,
				"latent_seq_len": 4,
				"latent_side":    2,
				"axis_count":     4,
				"axis_dim_0":     2,
				"axis_dim_1":     2,
				"axis_dim_2":     2,
				"axis_dim_3":     2,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one MultiAxisRoPE device call", func() {
			convey.So(len(recorder.multiAxisRoPECalls), convey.ShouldEqual, 1)

			call := recorder.multiAxisRoPECalls[0]
			convey.So(call.batch, convey.ShouldEqual, 1)
			convey.So(call.seqLen, convey.ShouldEqual, 8)
			convey.So(call.numHeads, convey.ShouldEqual, 2)
			convey.So(call.headDim, convey.ShouldEqual, 8)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
			convey.So(call.config.BaseFreq, convey.ShouldEqual, 2000.0)
			convey.So(call.config.LatentSeqLen, convey.ShouldEqual, 4)
			convey.So(call.config.LatentSide, convey.ShouldEqual, 2)
			convey.So(call.config.AxisCount, convey.ShouldEqual, 4)
			convey.So(call.config.AxisDim0, convey.ShouldEqual, 2)
			convey.So(call.config.AxisDim1, convey.ShouldEqual, 2)
			convey.So(call.config.AxisDim2, convey.ShouldEqual, 2)
			convey.So(call.config.AxisDim3, convey.ShouldEqual, 2)
		})

		convey.Convey("The output tensor shape follows the input", func() {
			stored, err := dispatcher.values.tensor("rope")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 8, 2, 8})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeUsesSDPABind(t *testing.T) {
	convey.Convey("Given attention.sdpa is declared with a YAML bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		query := uploadFloatSliceWithShape(t, memory, make([]float32, 1*8*2*4), []int{1, 8, 2, 4})
		key := uploadFloatSliceWithShape(t, memory, make([]float32, 1*8*2*4), []int{1, 8, 2, 4})
		value := uploadFloatSliceWithShape(t, memory, make([]float32, 1*8*2*4), []int{1, 8, 2, 4})
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("q", query)
		dispatcher.values.set("k", key)
		dispatcher.values.set("v", value)

		node := &ast.GraphNode{
			ID:     "attention",
			Op:     "attention.sdpa",
			Inputs: []string{"q", "k", "v"},
			Attributes: map[string]any{
				"num_heads": 2,
				"head_dim":  4,
				"causal":    false,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router issued one MultiHeadAttention device call", func() {
			convey.So(len(recorder.multiHeadCalls), convey.ShouldEqual, 1)

			call := recorder.multiHeadCalls[0]
			convey.So(call.config.NumHeads, convey.ShouldEqual, 2)
			convey.So(call.config.HeadDim, convey.ShouldEqual, 4)
			convey.So(call.config.Causal, convey.ShouldBeFalse)
			convey.So(call.config.KVHeadCount, convey.ShouldEqual, 2)
			convey.So(call.seqQ, convey.ShouldEqual, 8)
			convey.So(call.seqK, convey.ShouldEqual, 8)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor shape follows the query", func() {
			stored, err := dispatcher.values.tensor("attention")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 8, 2, 4})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunBoundNodeResolvesRoPEPositionFromDeviceTensor(t *testing.T) {
	convey.Convey("Given positional.rope carries a device-resident position tensor", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, make([]float32, 2*2*4), []int{2, 2, 4})
		position := newDispatchTestTensorWithRaw(
			t,
			[]int{1},
			dtype.Int32,
			unsafe.Pointer(uintptr(0x8000)),
			[]byte{9, 0, 0, 0},
		)
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.values.set("x", input)
		dispatcher.values.set("position", position)

		node := &ast.GraphNode{
			ID:     "rope",
			Op:     "positional.rope",
			Inputs: []string{"x", "position"},
			Attributes: map[string]any{
				"base":                  500000.0,
				"mode":                  "half",
				"rope_type":             "llama3",
				"rope_factor":           32.0,
				"rope_low_freq_factor":  1.0,
				"rope_high_freq_factor": 4.0,
				"rope_original_context": 8192,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router should pass the decoded start position", func() {
			convey.So(len(recorder.ropeCalls), convey.ShouldEqual, 1)
			convey.So(recorder.ropeCalls[0].config.StartPosition, convey.ShouldEqual, 9)
		})
	})
}

func TestRunBoundNodeResolvesRMSNormConfig(t *testing.T) {
	convey.Convey("Given math.rmsnorm carries a manifest epsilon", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, make([]float32, 2*4), []int{2, 4})
		scale := uploadFloatSlice(t, memory, []float32{1, 1, 1, 1})
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = mapWeightStore{"norm.weight": scale}
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "norm",
			Op:     "math.rmsnorm",
			Inputs: []string{"x"},
			Weights: &ast.BoundWeight{
				TensorName: "norm.weight",
			},
			Attributes: map[string]any{
				"eps": 1e-5,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The router passes the RMSNormConfig to the device", func() {
			convey.So(len(recorder.rmsNormCalls), convey.ShouldEqual, 1)
			convey.So(recorder.rmsNormCalls[0].config.Epsilon, convey.ShouldAlmostEqual, 1e-5)
			convey.So(recorder.rmsNormCalls[0].rows, convey.ShouldEqual, 2)
			convey.So(recorder.rmsNormCalls[0].lastDim, convey.ShouldEqual, 4)
			convey.So(recorder.rmsNormCalls[0].format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestOperationRegistrySelectsVariant(t *testing.T) {
	convey.Convey("Given activation.swiglu has packed and two-input variants", t, func() {
		registry, err := defaultOperationRegistry()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("One input selects the packed device method", func() {
			bind, err := registry.Bind(&ast.GraphNode{
				ID:     "packed",
				Op:     "activation.swiglu",
				Inputs: []string{"gate_up"},
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(bind.Method, convey.ShouldEqual, "SwiGLU")
		})

		convey.Convey("Two inputs select the tensor-pair device method", func() {
			bind, err := registry.Bind(&ast.GraphNode{
				ID:     "split",
				Op:     "activation.swiglu",
				Inputs: []string{"gate", "up"},
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(bind.Method, convey.ShouldEqual, "SwiGLUTensors")
		})
	})
}

func TestRunBoundNodeShapeIntrinsicWithLaunchBindings(t *testing.T) {
	convey.Convey("Given shape.last_token over a max-sized workspace view", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, []float32{
			1, 2, 3, 4,
			9, 9, 9, 9,
			5, 6, 7, 8,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
		}, []int{6, 4})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.maxBindings = ir.SymbolMap{"N": 6}
		dispatcher.launchBindings = ir.SymbolMap{"N": 3}
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "last",
			Op:     "shape.last_token",
			Inputs: []string{"x"},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The result selects the live final row", func() {
			stored, err := dispatcher.values.tensor("last")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 4})

			values, err := stored.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{5, 6, 7, 8})
		})
	})
}

func TestRunBoundNodeShapeIntrinsic(t *testing.T) {
	convey.Convey("Given shape.last_token is declared as an intrinsic bind", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, []float32{1, 2, 3, 4, 5, 6}, []int{3, 2})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "last",
			Op:     "shape.last_token",
			Inputs: []string{"x"},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The result contains the final row", func() {
			stored, err := dispatcher.values.tensor("last")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{1, 2})

			values, err := stored.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{5, 6})
		})

		convey.Convey("The result does not alias the consumed input row", func() {
			inputValues, err := input.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			inputValues[4] = 0
			inputValues[5] = 0

			stored, err := dispatcher.values.tensor("last")
			convey.So(err, convey.ShouldBeNil)

			values, err := stored.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{5, 6})
		})
	})
}

func TestCallRouterRejectsUnknownMethod(t *testing.T) {
	convey.Convey("Given an OperationBind with an unregistered method name", t, func() {
		bind := OperationBind{Method: "NotARealDeviceMethod"}

		err := callRouter(noopDeviceBackend{}, bind, nil, nil)

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "unknown method")
		convey.So(err.Error(), convey.ShouldContainSubstring, "NotARealDeviceMethod")
	})
}

func TestCallRouterRejectsWrongArgCount(t *testing.T) {
	convey.Convey("Given an Add bind with only three args", t, func() {
		err := callRouter(noopDeviceBackend{}, OperationBind{Method: "Add"}, nil, []any{
			unsafeNilPointer,
			unsafeNilPointer,
			unsafeNilPointer,
		})

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "Add expects 5 args")
	})
}

func TestResolveArgInputShape(t *testing.T) {
	convey.Convey("Given an input tensor with shape [2, 3, 4]", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
		}, []int{2, 3, 4})

		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		resolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "n",
				Op:     "test",
				Inputs: []string{"x"},
			},
			bind: OperationBind{InputNames: []string{"x"}},
		}

		convey.Convey("dim 0 resolves to 2", func() {
			value, err := resolver.resolveArg(asset.BindArg{
				Ref: "input.x.shape",
				Dim: intPointer(0),
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, 2)
		})

		convey.Convey("dim -1 resolves to the last dim", func() {
			value, err := resolver.resolveArg(asset.BindArg{
				Ref: "input.x.shape",
				Dim: intPointer(-1),
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, 4)
		})

		convey.Convey("drop_tail plus product resolves row count", func() {
			value, err := resolver.resolveArg(asset.BindArg{
				Ref:      "input.x.shape",
				DropTail: 1,
				Product:  true,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, 6)
		})
	})
}

func newTestDispatcher(deviceBackend executionDevice, memory tensor.Backend) *dispatcher {
	return &dispatcher{
		values:        newValueTable(),
		graph:         &ast.Graph{},
		graphName:     "test",
		deviceBackend: deviceBackend,
		memory:        memory,
		weights:       nilWeightStore{},
	}
}

type mapWeightStore map[string]tensor.Tensor

func (store mapWeightStore) Lookup(name string) (tensor.Tensor, error) {
	resident, ok := store[name]

	if !ok {
		return nil, ErrWeightNotFound
	}

	return resident, nil
}

func uploadFloatSlice(t testing.TB, memory tensor.Backend, values []float32) tensor.Tensor {
	t.Helper()

	return uploadFloatSliceWithShape(t, memory, values, []int{len(values)})
}

func uploadFloatSliceWithShape(t testing.TB, memory tensor.Backend, values []float32, dims []int) tensor.Tensor {
	t.Helper()

	shape, err := tensor.NewShape(dims)

	if err != nil {
		t.Fatalf("uploadFloatSliceWithShape: shape: %v", err)
	}

	bytes := make([]byte, len(values)*4)

	for index, value := range values {
		bits := *(*uint32)(unsafe.Pointer(&value))
		bytes[index*4] = byte(bits)
		bytes[index*4+1] = byte(bits >> 8)
		bytes[index*4+2] = byte(bits >> 16)
		bytes[index*4+3] = byte(bits >> 24)
	}

	resident, err := memory.Upload(shape, dtype.Float32, bytes)

	if err != nil {
		t.Fatalf("uploadFloatSliceWithShape: upload: %v", err)
	}

	return resident
}

func uploadInt32Slice(t testing.TB, memory tensor.Backend, values []int32) tensor.Tensor {
	t.Helper()

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		t.Fatalf("uploadInt32Slice: shape: %v", err)
	}

	bytes := make([]byte, len(values)*4)

	for index, value := range values {
		raw := uint32(value)
		bytes[index*4] = byte(raw)
		bytes[index*4+1] = byte(raw >> 8)
		bytes[index*4+2] = byte(raw >> 16)
		bytes[index*4+3] = byte(raw >> 24)
	}

	resident, err := memory.Upload(shape, dtype.Int32, bytes)

	if err != nil {
		t.Fatalf("uploadInt32Slice: upload: %v", err)
	}

	return resident
}

func intPointer(value int) *int {
	return &value
}
