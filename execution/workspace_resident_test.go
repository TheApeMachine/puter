package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestAttachResidentUsesPlannedWorkspaceSlots(testingObject *testing.T) {
	convey.Convey("Given a planned topology with reusable resident intervals", testingObject, func() {
		firstPort := workspaceResidentPort(1, 8)
		secondPort := workspaceResidentPort(2, 4)
		topology := &ir.Topology{
			Nodes: []*ir.Node{
				{Name: "first", Outputs: []*ir.Port{firstPort}},
				{Name: "second", Outputs: []*ir.Port{secondPort}},
			},
			Workspace: ir.WorkspaceLayout{
				Size:  64,
				Align: 64,
				Allocations: []ir.Interval{
					{PortID: 1, Start: 0, End: 0, Offset: 0, Size: 64},
					{PortID: 2, Start: 1, End: 1, Offset: 0, Size: 64},
				},
			},
		}
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{ID: "first", Op: "test.first"},
				{ID: "second", Op: "test.second"},
			},
		}
		backend := &recordingWorkspaceBackend{testingObject: testingObject}
		workspaceMap := NewWorkspaceMap()
		defer workspaceMap.Close()

		err := workspaceMap.AttachResident("model", graph, topology, backend)

		convey.Convey("It should allocate one resident slot and attach both outputs", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(backend.slotBytes, convey.ShouldResemble, []int{64})

			first, ok := workspaceMap.OutputFor("model", "first")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(first.Location(), convey.ShouldEqual, tensor.Metal)
			convey.So(first.Shape().Dims(), convey.ShouldResemble, []int{8})

			second, ok := workspaceMap.OutputFor("model", "second")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(second.Location(), convey.ShouldEqual, tensor.Metal)
			convey.So(second.Shape().Dims(), convey.ShouldResemble, []int{4})
		})
	})
}

func workspaceResidentPort(portID int32, count int64) *ir.Port {
	return &ir.Port{
		ID: portID,
		Type: ir.PortType{
			DType: dtype.Float32,
			ShapeSchema: ir.ShapeSchema{
				Dimensions: []ir.Dimension{{Static: count}},
			},
			Layout: ir.LayoutContiguous,
		},
	}
}

type recordingWorkspaceBackend struct {
	testingObject *testing.T
	slotBytes     []int
}

func (backend *recordingWorkspaceBackend) Location() tensor.Location {
	return tensor.Metal
}

func (backend *recordingWorkspaceBackend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{dtype.Float32, dtype.Int8}
}

func (backend *recordingWorkspaceBackend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

func (backend *recordingWorkspaceBackend) Capabilities() tensor.Capabilities {
	return tensor.Capabilities{NativeAlignment: 64}
}

func (backend *recordingWorkspaceBackend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	rawBytes []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (backend *recordingWorkspaceBackend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	rawBytes []byte,
) (tensor.Tensor, error) {
	return backend.Upload(shape, sourceDType, rawBytes)
}

func (backend *recordingWorkspaceBackend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (backend *recordingWorkspaceBackend) Download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

func (backend *recordingWorkspaceBackend) Close() error {
	return nil
}

func (backend *recordingWorkspaceBackend) AllocateWorkspaceSlot(byteCount int) (tensor.Tensor, error) {
	backend.slotBytes = append(backend.slotBytes, byteCount)

	return newDispatchTestTensor(
		backend.testingObject,
		[]int{byteCount},
		dtype.Int8,
		unsafe.Pointer(&backend.slotBytes[0]),
	), nil
}

func (backend *recordingWorkspaceBackend) ViewWorkspaceSlot(
	slot tensor.Tensor,
	shape tensor.Shape,
	elementFormat dtype.DType,
	byteCount int,
) (tensor.Tensor, error) {
	return newDispatchTestTensor(
		backend.testingObject,
		shape.Dims(),
		elementFormat,
		unsafe.Pointer(&backend.slotBytes[0]),
	), nil
}
