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
