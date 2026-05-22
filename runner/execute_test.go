package runner

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

type stubCommandBatcher struct {
	beginCount int
	endCount   int
}

func (batcher *stubCommandBatcher) BeginBatch() {
	batcher.beginCount++
}

func (batcher *stubCommandBatcher) EndBatch() error {
	batcher.endCount++
	return nil
}

type batchingBackend struct {
	*tensor.HostBackend
	stubCommandBatcher
}

func TestGraphCommandBatcherFor(testingObject *testing.T) {
	Convey("Given graphCommandBatcherFor", testingObject, func() {
		Convey("It should return nil for host memory", func() {
			hostMemory := tensor.NewHostBackend()

			So(graphCommandBatcherFor(tensor.Host, hostMemory), ShouldBeNil)
		})

		Convey("It should return a batcher for Metal backends that implement batching", func() {
			memory := &batchingBackend{HostBackend: tensor.NewHostBackend()}

			So(graphCommandBatcherFor(tensor.Metal, memory), ShouldNotBeNil)
		})
	})
}

func TestTensorUseCounts(testingObject *testing.T) {
	Convey("Given a compute graph with shared inputs", testingObject, func() {
		inputNode := ir.NewNode("input_ids", ir.OpInput, tensor.Shape{})
		leftNode := ir.NewNode("left", ir.OpType("math.add"), tensor.Shape{})
		rightNode := ir.NewNode("right", ir.OpType("math.add"), tensor.Shape{})

		leftNode.AddInput(inputNode)
		rightNode.AddInput(inputNode)

		computeGraph := ir.NewGraph()
		computeGraph.AddNode(inputNode)
		computeGraph.AddNode(leftNode)
		computeGraph.AddNode(rightNode)

		counts := tensorUseCounts(computeGraph)

		Convey("It should count every consumer edge", func() {
			So(counts["input_ids"], ShouldEqual, 2)
		})
	})
}

func TestReleaseConsumedTensors(testingObject *testing.T) {
	Convey("Given releaseConsumedTensors", testingObject, func() {
		hostMemory := tensor.NewHostBackend()
		tensorWorkspace := newWorkspace()

		shape, err := tensor.NewShape([]int{2, 2})

		So(err, ShouldBeNil)

		value, err := hostMemory.Upload(shape, dtype.Float32, []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})

		So(err, ShouldBeNil)

		tensorWorkspace.Store("shared", value)

		inputNode := ir.NewNode("shared", ir.OpInput, shape)
		consumer := ir.NewNode("consumer", ir.OpType("math.add"), shape)
		consumer.AddInput(inputNode)

		remainingUses := map[string]int{"shared": 1}

		releaseConsumedTensors(consumer, remainingUses, tensorWorkspace)

		Convey("It should release owned tensors when the last consumer finishes", func() {
			_, ok := tensorWorkspace.Load("shared")

			So(ok, ShouldBeFalse)
			So(remainingUses["shared"], ShouldEqual, 0)
		})
	})
}
