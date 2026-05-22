package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRopeConfigFromNode(t *testing.T) {
	convey.Convey("Given a positional.rope node with Llama 3 attributes", t, func() {
		shape, err := tensor.NewShape([]int{1, 32, 64})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("rope_q_0", "positional.rope", ir.OpFused, shape)
		node.SetAttribute("base", ir.FloatAttribute(500000))
		node.SetAttribute("rope_type", ir.StringAttribute("llama3"))
		node.SetAttribute("rope_factor", ir.FloatAttribute(32))
		node.SetAttribute("rope_low_freq_factor", ir.FloatAttribute(1))
		node.SetAttribute("rope_high_freq_factor", ir.FloatAttribute(4))
		node.SetAttribute("rope_original_context", ir.IntAttribute(8192))

		config := ropeConfigFromNode(node)

		convey.Convey("It should preserve base and llama3 scaling fields", func() {
			convey.So(config.Base, convey.ShouldEqual, float32(500000))
			convey.So(config.Type, convey.ShouldEqual, "llama3")
			convey.So(config.Factor, convey.ShouldEqual, float32(32))
			convey.So(config.LowFreqFactor, convey.ShouldEqual, float32(1))
			convey.So(config.HighFreqFactor, convey.ShouldEqual, float32(4))
			convey.So(config.OriginalContext, convey.ShouldEqual, uint32(8192))
		})
	})
}
