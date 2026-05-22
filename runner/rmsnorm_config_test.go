package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRMSNormEpsilonFromNode(t *testing.T) {
	convey.Convey("Given a math.rmsnorm node with eps attribute", t, func() {
		shape, err := tensor.NewShape([]int{1, 32, 2048})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("norm1_0", "math.rmsnorm", ir.OpFused, shape)
		node.SetAttribute("eps", ir.FloatAttribute(1e-5))

		convey.Convey("It should preserve the configured epsilon", func() {
			convey.So(rmsNormEpsilonFromNode(node), convey.ShouldEqual, float32(1e-5))
		})
	})

	convey.Convey("Given a math.rmsnorm node without eps", t, func() {
		shape, err := tensor.NewShape([]int{1, 32, 2048})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("norm1_0", "math.rmsnorm", ir.OpFused, shape)

		convey.Convey("It should use the Metal default epsilon", func() {
			convey.So(rmsNormEpsilonFromNode(node), convey.ShouldEqual, float32(1e-6))
		})
	})
}
