package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunUpsampleNearest2DIntrinsicCopiesHostNCHW(testingObject *testing.T) {
	convey.Convey("Given a host NCHW tensor for nearest upsampling", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2,
			3, 4,
		}, []int{1, 1, 2, 2})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "up",
			Op:     "shape.upsample_nearest2d",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"scale_h": 2,
				"scale_w": 2,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should replicate each input pixel into a 2x2 block", func() {
			output, err := dispatcher.values.tensor("up")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{1, 1, 4, 4})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{
				1, 1, 2, 2,
				1, 1, 2, 2,
				3, 3, 4, 4,
				3, 3, 4, 4,
			})
		})
	})
}

func TestRunUpsampleNearest2DIntrinsicDispatchesDeviceNCHW(testingObject *testing.T) {
	convey.Convey("Given a device NCHW tensor for nearest upsampling", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputPointer := unsafe.Pointer(uintptr(0x9100))
		outputPointer := unsafe.Pointer(uintptr(0x9200))
		input := newDispatchTestTensor(testingObject, []int{2, 3, 5, 7}, dtype.Float32, inputPointer)
		output := newDispatchTestTensor(testingObject, []int{2, 3, 10, 14}, dtype.Float32, outputPointer)

		deviceBackend := &recordingUpsampleNearest2DDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)
		dispatcher.workspaces = &WorkspaceMap{
			outputs: map[string]map[string]tensor.Tensor{
				"test": {"up": output},
			},
		}
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "up",
			Op:     "shape.upsample_nearest2d",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"scale_h": 2,
				"scale_w": 2,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should call the device upsample hook with NCHW dimensions", func() {
			convey.So(len(deviceBackend.calls), convey.ShouldEqual, 1)

			call := deviceBackend.calls[0]
			convey.So(call.input, convey.ShouldEqual, inputPointer)
			convey.So(call.output, convey.ShouldEqual, outputPointer)
			convey.So(call.channels, convey.ShouldEqual, 3)
			convey.So(call.inHeight, convey.ShouldEqual, 5)
			convey.So(call.inWidth, convey.ShouldEqual, 7)
			convey.So(call.outHeight, convey.ShouldEqual, 10)
			convey.So(call.outWidth, convey.ShouldEqual, 14)
			convey.So(call.outElements, convey.ShouldEqual, 2*3*10*14)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type upsampleNearest2DCall struct {
	input       unsafe.Pointer
	output      unsafe.Pointer
	channels    int
	inHeight    int
	inWidth     int
	outHeight   int
	outWidth    int
	outElements int
	format      dtype.DType
}

type recordingUpsampleNearest2DDevice struct {
	noopDeviceBackend
	calls []upsampleNearest2DCall
}

func (recorder *recordingUpsampleNearest2DDevice) IntrinsicUpsampleNearest2D(
	input, output unsafe.Pointer,
	channels, inHeight, inWidth, outHeight, outWidth, outElements int,
	format dtype.DType,
) {
	recorder.calls = append(recorder.calls, upsampleNearest2DCall{
		input:       input,
		output:      output,
		channels:    channels,
		inHeight:    inHeight,
		inWidth:     inWidth,
		outHeight:   outHeight,
		outWidth:    outWidth,
		outElements: outElements,
		format:      format,
	})
}
