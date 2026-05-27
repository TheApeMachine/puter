package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunLastTokenIntrinsicDispatchesDeviceInput(testingObject *testing.T) {
	convey.Convey("Given device-resident sequence activations", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputPointer := unsafe.Pointer(uintptr(0x7000))
		input := newDispatchTestTensor(testingObject, []int{4, 3}, dtype.Float32, inputPointer)
		deviceBackend := &recordingLastTokenDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)

		dispatcher.values.set("hidden", input)

		outputShape, err := tensor.NewShape([]int{3})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: outputShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "last",
				Op:     "shape.last_token",
				Inputs: []string{"hidden"},
			},
		}

		convey.Convey("It should call the last-token device hook", func() {
			output, err := runLastTokenIntrinsic(resolver)

			convey.So(err, convey.ShouldBeNil)
			convey.So(output, convey.ShouldNotBeNil)
			convey.So(deviceBackend.call.input, convey.ShouldEqual, inputPointer)
			convey.So(deviceBackend.call.seq, convey.ShouldEqual, 4)
			convey.So(deviceBackend.call.hiddenBytes, convey.ShouldEqual, 12)
			convey.So(deviceBackend.call.outBytes, convey.ShouldEqual, 12)
			convey.So(deviceBackend.call.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunLastTokenIntrinsicCopiesBatchedHostInput(testingObject *testing.T) {
	convey.Convey("Given batched host sequence activations", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2,
			3, 4,
			5, 6,
			10, 20,
			30, 40,
			50, 60,
		}, []int{2, 3, 2})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("hidden", input)

		node := &ast.GraphNode{
			ID:     "last",
			Op:     "shape.last_token",
			Inputs: []string{"hidden"},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should copy the final sequence row for each batch", func() {
			output, err := dispatcher.values.tensor("last")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{2, 2})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{5, 6, 50, 60})
		})
	})
}

type lastTokenCall struct {
	input       unsafe.Pointer
	output      unsafe.Pointer
	seq         int
	hiddenBytes int
	outBytes    int
	format      dtype.DType
}

type recordingLastTokenDevice struct {
	noopDeviceBackend
	call lastTokenCall
}

func (recorder *recordingLastTokenDevice) LastToken(
	input, output unsafe.Pointer,
	seq, hiddenBytes, outBytes int,
	format dtype.DType,
) {
	recorder.call = lastTokenCall{
		input:       input,
		output:      output,
		seq:         seq,
		hiddenBytes: hiddenBytes,
		outBytes:    outBytes,
		format:      format,
	}
}

func TestRunConcatIntrinsicCopiesHostLastDim(testingObject *testing.T) {
	convey.Convey("Given host tensors concatenated along the last dimension", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		left := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2,
			3, 4,
		}, []int{1, 2, 2})
		right := uploadFloatSliceWithShape(testingObject, memory, []float32{
			10,
			20,
		}, []int{1, 2, 1})

		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("left", left)
		dispatcher.values.set("right", right)

		node := &ast.GraphNode{
			ID:     "joined",
			Op:     "shape.concat",
			Inputs: []string{"left", "right"},
			Attributes: map[string]any{
				"dim": 2,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should interleave each row's right-hand tail", func() {
			output, err := dispatcher.values.tensor("joined")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{1, 2, 3})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{1, 2, 10, 3, 4, 20})
		})
	})
}

func TestRunConcatIntrinsicDispatchesMetalLastDim(testingObject *testing.T) {
	convey.Convey("Given Metal tensors concatenated along the last dimension", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		leftPointer := unsafe.Pointer(uintptr(0x7100))
		rightPointer := unsafe.Pointer(uintptr(0x7200))
		outputPointer := unsafe.Pointer(uintptr(0x7300))
		left := newDispatchTestTensor(testingObject, []int{1, 2, 2}, dtype.Float32, leftPointer)
		right := newDispatchTestTensor(testingObject, []int{1, 2, 1}, dtype.Float32, rightPointer)
		output := newDispatchTestTensor(testingObject, []int{1, 2, 3}, dtype.Float32, outputPointer)

		deviceBackend := &recordingConcatDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)
		dispatcher.workspaces = &WorkspaceMap{
			outputs: map[string]map[string]tensor.Tensor{
				"test": {
					"joined": output,
				},
			},
		}
		dispatcher.values.set("left", left)
		dispatcher.values.set("right", right)

		node := &ast.GraphNode{
			ID:     "joined",
			Op:     "shape.concat",
			Inputs: []string{"left", "right"},
			Attributes: map[string]any{
				"dim": 2,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should call the last-dimension Metal hook", func() {
			convey.So(len(deviceBackend.lastDimCalls), convey.ShouldEqual, 1)

			call := deviceBackend.lastDimCalls[0]
			convey.So(call.left, convey.ShouldEqual, leftPointer)
			convey.So(call.right, convey.ShouldEqual, rightPointer)
			convey.So(call.output, convey.ShouldEqual, outputPointer)
			convey.So(call.leftRowBytes, convey.ShouldEqual, 8)
			convey.So(call.rightRowBytes, convey.ShouldEqual, 4)
			convey.So(call.rowBytes, convey.ShouldEqual, 12)
			convey.So(call.totalBytes, convey.ShouldEqual, 24)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type concatCall struct {
	left          unsafe.Pointer
	right         unsafe.Pointer
	output        unsafe.Pointer
	leftBytes     int
	rightBytes    int
	leftRowBytes  int
	rightRowBytes int
	rowBytes      int
	totalBytes    int
	format        dtype.DType
}

type recordingConcatDevice struct {
	noopDeviceBackend
	contiguousCalls []concatCall
	lastDimCalls    []concatCall
}

func (recorder *recordingConcatDevice) Concat(
	left, right, output unsafe.Pointer,
	leftBytes, rightBytes int,
	format dtype.DType,
) {
	recorder.contiguousCalls = append(recorder.contiguousCalls, concatCall{
		left:       left,
		right:      right,
		output:     output,
		leftBytes:  leftBytes,
		rightBytes: rightBytes,
		format:     format,
	})
}

func (recorder *recordingConcatDevice) ConcatLastDim(
	left, right, output unsafe.Pointer,
	leftRowBytes, rightRowBytes, rowBytes, totalBytes int,
	format dtype.DType,
) {
	recorder.lastDimCalls = append(recorder.lastDimCalls, concatCall{
		left:          left,
		right:         right,
		output:        output,
		leftRowBytes:  leftRowBytes,
		rightRowBytes: rightRowBytes,
		rowBytes:      rowBytes,
		totalBytes:    totalBytes,
		format:        format,
	})
}
