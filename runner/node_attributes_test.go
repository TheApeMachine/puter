package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestNodeOptionalFloatAttribute(testingObject *testing.T) {
	convey.Convey("Given integer IR attributes", testingObject, func() {
		shape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("rope_q_0", "positional.rope", ir.OpFused, shape)
		node.SetAttribute("base", ir.IntAttribute(500000))

		value, ok := nodeOptionalFloatAttribute(node, "base")

		convey.Convey("It should parse them as float values", func() {
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(value, convey.ShouldEqual, 500000)
		})
	})
}
