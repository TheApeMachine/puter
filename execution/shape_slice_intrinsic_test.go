package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunSliceIntrinsicCopiesHostMiddleDim(testingObject *testing.T) {
	convey.Convey("Given a host tensor sliced across a middle dimension", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2,
			3, 4,
			5, 6,
			7, 8,
			9, 10,
			11, 12,
			13, 14,
			15, 16,
		}, []int{2, 4, 2})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "sliced",
			Op:     "shape.slice",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"dim":   1,
				"start": 1,
				"end":   0,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should copy every selected row block", func() {
			output, err := dispatcher.values.tensor("sliced")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{2, 3, 2})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{
				3, 4,
				5, 6,
				7, 8,
				11, 12,
				13, 14,
				15, 16,
			})
		})
	})
}

func TestRunSliceIntrinsicDispatchesDeviceMiddleDim(testingObject *testing.T) {
	convey.Convey("Given a device tensor sliced across a middle dimension", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputPointer := unsafe.Pointer(uintptr(0x8100))
		outputPointer := unsafe.Pointer(uintptr(0x8200))
		input := newDispatchTestTensor(testingObject, []int{2, 4, 2}, dtype.Float32, inputPointer)
		output := newDispatchTestTensor(testingObject, []int{2, 2, 2}, dtype.Float32, outputPointer)

		deviceBackend := &recordingSliceDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)
		dispatcher.workspaces = &WorkspaceMap{
			outputs: map[string]map[string]tensor.Tensor{
				"test": {"sliced": output},
			},
		}
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "sliced",
			Op:     "shape.slice",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"dim":   1,
				"start": 1,
				"end":   3,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should call the device slice hook with row-major strides", func() {
			convey.So(len(deviceBackend.calls), convey.ShouldEqual, 1)

			call := deviceBackend.calls[0]
			convey.So(call.input, convey.ShouldEqual, inputPointer)
			convey.So(call.output, convey.ShouldEqual, outputPointer)
			convey.So(call.sliceLen, convey.ShouldEqual, 2)
			convey.So(call.inputDimSize, convey.ShouldEqual, 4)
			convey.So(call.innerBytes, convey.ShouldEqual, 8)
			convey.So(call.start, convey.ShouldEqual, 1)
			convey.So(call.outBytes, convey.ShouldEqual, 32)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type sliceCall struct {
	input        unsafe.Pointer
	output       unsafe.Pointer
	sliceLen     int
	inputDimSize int
	innerBytes   int
	start        int
	outBytes     int
	format       dtype.DType
}

type recordingSliceDevice struct {
	noopDeviceBackend
	calls []sliceCall
}

func (recorder *recordingSliceDevice) Slice(
	input, output unsafe.Pointer,
	sliceLen, inputDimSize, innerBytes, start, outBytes int,
	format dtype.DType,
) {
	recorder.calls = append(recorder.calls, sliceCall{
		input:        input,
		output:       output,
		sliceLen:     sliceLen,
		inputDimSize: inputDimSize,
		innerBytes:   innerBytes,
		start:        start,
		outBytes:     outBytes,
		format:       format,
	})
}
