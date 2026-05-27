package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

type recordingDevice struct {
	addCalls          []recordedAddCall
	rmsNormCalls      []recordedRMSNormCall
	swiGLUTensorCalls []recordedSwiGLUTensorCall
	ropeCalls         []recordedRoPECall
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

type recordedRMSNormCall struct {
	config  device.RMSNormConfig
	input   unsafe.Pointer
	scale   unsafe.Pointer
	output  unsafe.Pointer
	rows    int
	lastDim int
	format  dtype.DType
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

func (recordingDevice) LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("recordingDevice.LayerNorm invoked")
}

func (recordingDevice) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	panic("recordingDevice.Matmul invoked")
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

func (recordingDevice) MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	panic("recordingDevice.MultiHeadAttention invoked")
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

		input := uploadFloatSliceWithShape(t, memory, make([]float32, 2*2*4), []int{2, 2, 4})
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
		plan:          &runtime.ExecutionPlan{},
		nodeByID:      make(map[string]*ast.GraphNode),
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

func uploadFloatSlice(t *testing.T, memory tensor.Backend, values []float32) tensor.Tensor {
	t.Helper()

	return uploadFloatSliceWithShape(t, memory, values, []int{len(values)})
}

func uploadFloatSliceWithShape(t *testing.T, memory tensor.Backend, values []float32, dims []int) tensor.Tensor {
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

func uploadInt32Slice(t *testing.T, memory tensor.Backend, values []int32) tensor.Tensor {
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
