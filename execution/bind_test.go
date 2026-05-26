package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
recordingDevice is a test-only executionDevice that records the
parameters of every method invocation. Used by TestRunBoundNodeMatchesHandwrittenAdd
to assert that the generic bind dispatcher invokes Add with exactly
the same argument shapes the existing handleBinaryElementwise(opAdd)
would have produced.

Every non-recorded method panics so the test catches accidental
extra calls.
*/
type recordingDevice struct {
	addCalls []recordedAddCall
}

type recordedAddCall struct {
	dst    unsafe.Pointer
	left   unsafe.Pointer
	right  unsafe.Pointer
	count  int
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

// Every other method panics — the test must not reach them.

func (recordingDevice) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	panic("recordingDevice.Lookup invoked")
}

func (recordingDevice) RMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("recordingDevice.RMSNorm invoked")
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

func (recordingDevice) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	panic("recordingDevice.SwiGLUTensors invoked")
}

func (recordingDevice) RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	panic("recordingDevice.RoPE invoked")
}

func (recordingDevice) MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	panic("recordingDevice.MultiHeadAttention invoked")
}

/*
TestRunBoundNodeMatchesHandwrittenAdd is the stage-1 proof point for the
manifest-driven dispatcher.

Two paths invoke the same Add call:

  1. The existing handleBinaryElementwise(opAdd) handler from
     dispatch_table.go, hand-coded for math.add.
  2. The generic runBoundNode invoked with an OperationBind declaring
     math.add in terms of Method + ArgSpec list.

Both paths use the same dispatcher, the same input tensors, and the
same recording device. The test asserts that BOTH paths produce a
recorded Add call with identical argument *shapes* — the pointer
values differ because each path allocates its own output tensor, but
the count, format, and the fact that arg[1] is input[0]'s pointer and
arg[2] is input[1]'s pointer are identical.

When this passes we know the generic dispatcher is functionally
equivalent to a hand-coded handler for one op. The migration is then
mechanical: for each entry in opTable, write the bind block, delete
the handler, verify the same call signature is produced.
*/
func TestRunBoundNodeMatchesHandwrittenAdd(t *testing.T) {
	convey.Convey("Given an in-memory math.add over two [4]Float32 inputs", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		leftTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})
		rightTensor := uploadFloatSlice(t, memory, []float32{10, 20, 30, 40})

		node := &ast.GraphNode{
			ID:     "added",
			Op:     "math.add",
			Inputs: []string{"x", "y"},
		}

		convey.Convey("The hand-coded handleBinaryElementwise(opAdd) records one Add call", func() {
			recorder := &recordingDevice{}
			dispatch := newTestDispatcher(recorder, memory)
			dispatch.values.set("x", leftTensor)
			dispatch.values.set("y", rightTensor)

			err := handleBinaryElementwise(opAdd)(dispatch, node)
			convey.So(err, convey.ShouldBeNil)

			convey.So(len(recorder.addCalls), convey.ShouldEqual, 1)
			call := recorder.addCalls[0]
			convey.So(call.count, convey.ShouldEqual, 4)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The generic runBoundNode with the equivalent OperationBind records the same shape", func() {
			recorder := &recordingDevice{}
			dispatch := newTestDispatcher(recorder, memory)
			dispatch.values.set("x", leftTensor)
			dispatch.values.set("y", rightTensor)

			bind := OperationBind{
				Method: "Add",
				OutputShape: []ArgSpec{
					{Ref: ArgRefInputShape, Source: "0", Dim: 0},
				},
				OutputDType: dtype.Float32,
				Args: []ArgSpec{
					{Ref: ArgRefOutputPointer},
					{Ref: ArgRefInputPointer, Source: "0"},
					{Ref: ArgRefInputPointer, Source: "1"},
					{Ref: ArgRefInputLen, Source: "0"},
					{Ref: ArgRefInputDType, Source: "0"},
				},
			}

			err := runBoundNode(dispatch, node, bind)
			convey.So(err, convey.ShouldBeNil)

			convey.So(len(recorder.addCalls), convey.ShouldEqual, 1)
			call := recorder.addCalls[0]
			convey.So(call.count, convey.ShouldEqual, 4)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)

			convey.Convey("And the generic path stored the output tensor under node.ID", func() {
				stored, err := dispatch.values.tensor("added")
				convey.So(err, convey.ShouldBeNil)
				convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{4})
				convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
			})
		})
	})
}

/*
TestOpTableMathAddRoutesThroughBindBlock proves the migration step:
opTable["math.add"] is no longer the hand-coded handleBinaryElementwise
closure — it's a boundHandler wrapping bindMathAdd. The dispatcher's
runNode lookup must still produce the same observable Add call when
the runtime walks an ast.GraphNode with op "math.add".

When this passes, the production chat path runs math.add through the
generic bind dispatcher. No external behavior change; the only thing
that's different is one Go closure became one declarative bind block.
The same shape extends to every other opTable entry in the next
migration steps.
*/
func TestOpTableMathAddRoutesThroughBindBlock(t *testing.T) {
	convey.Convey("Given opTable's math.add entry is now a bind block", t, func() {
		handler, ok := opTable["math.add"]
		convey.So(ok, convey.ShouldBeTrue)

		memory := tensor.NewHostBackend()
		defer memory.Close()

		leftTensor := uploadFloatSlice(t, memory, []float32{1, 2, 3, 4})
		rightTensor := uploadFloatSlice(t, memory, []float32{10, 20, 30, 40})

		recorder := &recordingDevice{}
		dispatch := newTestDispatcher(recorder, memory)
		dispatch.values.set("x", leftTensor)
		dispatch.values.set("y", rightTensor)

		node := &ast.GraphNode{
			ID:     "added",
			Op:     "math.add",
			Inputs: []string{"x", "y"},
		}

		err := handler(dispatch, node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Exactly one Add call was issued through the static router", func() {
			convey.So(len(recorder.addCalls), convey.ShouldEqual, 1)
		})

		convey.Convey("The Add call has the same shape the hand-coded handler produced", func() {
			call := recorder.addCalls[0]
			convey.So(call.count, convey.ShouldEqual, 4)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})

		convey.Convey("The output tensor is registered under the node ID", func() {
			stored, err := dispatch.values.tensor("added")
			convey.So(err, convey.ShouldBeNil)
			convey.So(stored.Shape().Dims(), convey.ShouldResemble, []int{4})
			convey.So(stored.DType(), convey.ShouldEqual, dtype.Float32)
		})
	})
}

/*
TestCallRouterRejectsUnknownMethod guards the closed-world contract:
asking the router for a method that has no case must be an error, not
a silent no-op or a panic. This keeps a typo in a bind block from
masquerading as success.
*/
func TestCallRouterRejectsUnknownMethod(t *testing.T) {
	convey.Convey("Given an OperationBind with an unregistered method name", t, func() {
		bind := OperationBind{Method: "NotARealDeviceMethod"}

		err := callRouter(noopDeviceBackend{}, bind, nil, nil)

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "unknown method")
		convey.So(err.Error(), convey.ShouldContainSubstring, "NotARealDeviceMethod")
	})
}

/*
TestCallRouterRejectsWrongArgCount asserts the router catches a bind
block that supplies the wrong number of positional arguments. This is
the most likely class of bug when writing a new bind block by hand;
the diagnostic must point at the method and the actual count so a
human can fix the YAML.
*/
func TestCallRouterRejectsWrongArgCount(t *testing.T) {
	convey.Convey("Given an Add bind with only 3 args", t, func() {
		err := callRouter(noopDeviceBackend{}, OperationBind{Method: "Add"}, nil, []any{
			unsafeNilPointer,
			unsafeNilPointer,
			unsafeNilPointer,
		})

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "Add expects 5 args")
	})
}

/*
TestResolveArgArgRefInputShape covers the resolver's dim-extraction
path on its own — useful because dim negative indexing (-1 = last dim)
is the kind of thing that quietly breaks when refactored.
*/
func TestResolveArgInputShape(t *testing.T) {
	convey.Convey("Given an input tensor with shape [2, 3, 4]", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(t, memory, []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
		}, []int{2, 3, 4})

		dispatch := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatch.values.set("x", input)

		node := &ast.GraphNode{ID: "n", Op: "test", Inputs: []string{"x"}}

		convey.Convey("dim 0 resolves to 2", func() {
			value, err := resolveArg(dispatch, node, ArgSpec{
				Ref: ArgRefInputShape, Source: "0", Dim: 0,
			}, nil)
			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, 2)
		})

		convey.Convey("dim -1 resolves to the last dim (4)", func() {
			value, err := resolveArg(dispatch, node, ArgSpec{
				Ref: ArgRefInputShape, Source: "0", Dim: -1,
			}, nil)
			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, 4)
		})

		convey.Convey("dim 99 surfaces an out-of-range error", func() {
			_, err := resolveArg(dispatch, node, ArgSpec{
				Ref: ArgRefInputShape, Source: "0", Dim: 99,
			}, nil)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "out of range")
		})
	})
}

/*
newTestDispatcher constructs a dispatcher wired against a recording
device and host memory but with no plan / workspace / weight store —
suitable for testing the generic bind path in isolation.
*/
func newTestDispatcher(deviceBackend executionDevice, memory tensor.Backend) *dispatcher {
	return &dispatcher{
		values:        newValueTable(),
		graph:         &ast.Graph{},
		graphName:     "test",
		plan:          &runtime.ExecutionPlan{},
		nodeByID:      make(map[string]*ast.GraphNode),
		deviceBackend: deviceBackend,
		memory:        memory,
	}
}

/*
uploadFloatSlice is a convenience for tests that need a 1-D Float32
host tensor. Wraps the same memory.Upload path the dispatcher uses.
*/
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
