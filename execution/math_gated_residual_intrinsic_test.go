package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunGatedResidualIntrinsicCopiesHostInput(testingObject *testing.T) {
	convey.Convey("Given host tensors feeding gated residual", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		residual := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		}, []int{2, 2, 3})
		branch := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1,
			1, 1, 1,
		}, []int{2, 2, 3})
		modulation := uploadFloatSliceWithShape(testingObject, memory, []float32{
			0, 0, 0, 0, 0, 0, 0.5, 1, 2,
			0, 0, 0, 0, 0, 0, 3, 4, 5,
		}, []int{2, 9})

		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("residual", residual)
		dispatcher.values.set("branch", branch)
		dispatcher.values.set("modulation", modulation)

		node := &ast.GraphNode{
			ID:     "merged",
			Op:     "math.gated_residual",
			Inputs: []string{"residual", "branch", "modulation"},
			Attributes: map[string]any{
				"set": 0,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should add the gated branch to every row", func() {
			output, err := dispatcher.values.tensor("merged")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{2, 2, 3})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{
				1.5, 3, 5,
				4.5, 6, 8,
				10, 12, 14,
				13, 15, 17,
			})
		})
	})
}

func TestRunGatedResidualIntrinsicDispatchesDeviceInput(testingObject *testing.T) {
	convey.Convey("Given device tensors feeding gated residual", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		residualPointer := unsafe.Pointer(uintptr(0x9100))
		branchPointer := unsafe.Pointer(uintptr(0x9200))
		modulationPointer := unsafe.Pointer(uintptr(0x9300))
		outputPointer := unsafe.Pointer(uintptr(0x9400))
		residual := newDispatchTestTensor(testingObject, []int{2, 4, 3}, dtype.Float32, residualPointer)
		branch := newDispatchTestTensor(testingObject, []int{2, 4, 3}, dtype.Float32, branchPointer)
		modulation := newDispatchTestTensor(testingObject, []int{2, 18}, dtype.Float32, modulationPointer)
		output := newDispatchTestTensor(testingObject, []int{2, 4, 3}, dtype.Float32, outputPointer)

		deviceBackend := &recordingGatedResidualDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)
		dispatcher.workspaces = &WorkspaceMap{
			outputs: map[string]map[string]tensor.Tensor{
				"test": {"merged": output},
			},
		}
		dispatcher.values.set("residual", residual)
		dispatcher.values.set("branch", branch)
		dispatcher.values.set("modulation", modulation)

		node := &ast.GraphNode{
			ID:     "merged",
			Op:     "math.gated_residual",
			Inputs: []string{"residual", "branch", "modulation"},
			Attributes: map[string]any{
				"set": 1,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should call the device hook with modulation layout", func() {
			convey.So(len(deviceBackend.calls), convey.ShouldEqual, 1)

			call := deviceBackend.calls[0]
			convey.So(call.residual, convey.ShouldEqual, residualPointer)
			convey.So(call.branch, convey.ShouldEqual, branchPointer)
			convey.So(call.modulation, convey.ShouldEqual, modulationPointer)
			convey.So(call.output, convey.ShouldEqual, outputPointer)
			convey.So(call.rows, convey.ShouldEqual, 8)
			convey.So(call.lastDim, convey.ShouldEqual, 3)
			convey.So(call.rowsPerBatch, convey.ShouldEqual, 4)
			convey.So(call.modulationCols, convey.ShouldEqual, 18)
			convey.So(call.set, convey.ShouldEqual, 1)
			convey.So(call.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type gatedResidualCall struct {
	residual       unsafe.Pointer
	branch         unsafe.Pointer
	modulation     unsafe.Pointer
	output         unsafe.Pointer
	rows           int
	lastDim        int
	rowsPerBatch   int
	modulationCols int
	set            int
	format         dtype.DType
}

type recordingGatedResidualDevice struct {
	noopDeviceBackend
	calls []gatedResidualCall
}

func (recorder *recordingGatedResidualDevice) GatedResidual(
	residual, branch, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols, set int,
	format dtype.DType,
) {
	recorder.calls = append(recorder.calls, gatedResidualCall{
		residual:       residual,
		branch:         branch,
		modulation:     modulation,
		output:         output,
		rows:           rows,
		lastDim:        lastDim,
		rowsPerBatch:   rowsPerBatch,
		modulationCols: modulationCols,
		set:            set,
		format:         format,
	})
}
