package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestBiasFeatureCount(testingObject *testing.T) {
	convey.Convey("Given a projection.linear node with out_features", testingObject, func() {
		shape, err := tensor.NewShape([]int{1, 1, 4096})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("q_proj", "projection.linear", ir.OpFused, shape)
		node.SetAttribute("out_features", ir.IntAttribute(4096))

		convey.Convey("It should size zero bias from out_features", func() {
			featureCount, err := biasFeatureCount(node)

			convey.So(err, convey.ShouldBeNil)
			convey.So(featureCount, convey.ShouldEqual, 4096)
		})
	})

	convey.Convey("Given a conv node with out_channels", testingObject, func() {
		shape, err := tensor.NewShape([]int{1, 64, 32, 32})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("conv", "conv2d", ir.OpFused, shape)
		node.SetAttribute("out_channels", ir.IntAttribute(64))

		convey.Convey("It should size zero bias from out_channels", func() {
			featureCount, err := biasFeatureCount(node)

			convey.So(err, convey.ShouldBeNil)
			convey.So(featureCount, convey.ShouldEqual, 64)
		})
	})
}
